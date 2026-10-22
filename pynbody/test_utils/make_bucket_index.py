"""Generate an index.html listing the contents of an R2 data bucket, and upload it to the bucket

This is run by .github/workflows/bucket-index.yaml using the R2 credentials stored as repository secrets.
That workflow is also called by other repositories (e.g. tangos) to index their own buckets, passing
--title and --description to replace the pynbody text. It requires boto3, and reads the following
environment variables:

  R2_ACCOUNT_ID, R2_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

To preview the page without uploading it, pass --output index.html.

Like the rest of this package's download utilities, it must not depend on a working pynbody installation.
"""

import argparse
import datetime
import html
import os
import urllib.parse

INDEX_KEY = 'index.html'

DEFAULT_TITLE = 'pynbody data'

DEFAULT_DESCRIPTION = """Test data and ion fraction tables for <a href="https://github.com/pynbody/pynbody">pynbody</a>.
You don't normally need to download these by hand: pynbody fetches them when needed, and
<code>pynbody.test_utils.precache_test_data()</code> downloads all the test data at once. See
<a href="https://pynbody.readthedocs.io/latest/tutorials/tutorials.html#obtaining-test-data">obtaining test data</a>."""

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
  th, td {{ text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid #8884; }}
  td.num, th.num {{ text-align: right; white-space: nowrap; width: 7rem; }}
  code {{ background: #8882; padding: 0 0.2rem; border-radius: 3px; }}
  .meta {{ color: #888; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>{description}</p>
<p class="meta">{num_files} files, {total_size} in total. Generated {generated}.</p>
{sections}
</body>
</html>
"""


def format_size(num_bytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if num_bytes < 1000 or unit == 'GB':
            return f"{num_bytes:.0f} {unit}" if unit == 'B' else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1000


def list_objects(client, bucket):
    """Return a list of (key, size, last_modified) for every object in the bucket except the index itself"""
    objects = []
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket):
        for obj in page.get('Contents', []):
            if obj['Key'] != INDEX_KEY:
                objects.append((obj['Key'], obj['Size'], obj['LastModified']))
    return objects


def render_index(objects, title=DEFAULT_TITLE, description=DEFAULT_DESCRIPTION):
    """Return the index page as a string, with one table per top-level folder

    The title is plain text; the description is inserted as HTML.
    """
    sections = {}
    for key, size, last_modified in sorted(objects):
        folder = key.split('/')[0] if '/' in key else ''
        sections.setdefault(folder, []).append((key, size, last_modified))

    sections_html = []
    for folder in sorted(sections, key=lambda f: (f == '', f)):
        rows = []
        for key, size, last_modified in sections[folder]:
            name = key[len(folder) + 1:] if folder else key
            rows.append(f'<tr><td><a href="{html.escape(urllib.parse.quote(key))}">{html.escape(name)}</a></td>'
                        f'<td class="num">{format_size(size)}</td>'
                        f'<td class="num">{last_modified:%Y-%m-%d}</td></tr>')
        folder_size = format_size(sum(size for _, size, _ in sections[folder]))
        heading = f"{html.escape(folder)}/" if folder else "Other files"
        sections_html.append(f'<h2>{heading} <span class="meta">({folder_size})</span></h2>\n'
                             '<table>\n<tr><th>File</th><th class="num">Size</th><th class="num">Modified</th></tr>\n'
                             + "\n".join(rows) + "\n</table>")

    return _PAGE_TEMPLATE.format(title=html.escape(title), description=description,
                                 num_files=len(objects),
                                 total_size=format_size(sum(size for _, size, _ in objects)),
                                 generated=f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC",
                                 sections="\n".join(sections_html))


def main():
    parser = argparse.ArgumentParser(description='Generate and upload an index.html for an R2 data bucket')
    parser.add_argument('--output', help='Write the page to this local file instead of uploading it')
    parser.add_argument('--title', help=f'Page title (default: {DEFAULT_TITLE!r}); an empty value also gives the default')
    parser.add_argument('--description', help='HTML paragraph introducing the data (default: the pynbody text); '
                                              'an empty value also gives the default')
    args = parser.parse_args()

    import boto3
    bucket = os.environ['R2_BUCKET']
    client = boto3.client('s3', region_name='auto',
                          endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com")

    objects = list_objects(client, bucket)
    page = render_index(objects, args.title or DEFAULT_TITLE, args.description or DEFAULT_DESCRIPTION)
    print(f"Listed {len(objects)} objects")

    if args.output:
        with open(args.output, 'w') as f:
            f.write(page)
        print(f"Wrote {args.output}")
    else:
        client.put_object(Bucket=bucket, Key=INDEX_KEY, Body=page.encode('utf-8'),
                          ContentType='text/html; charset=utf-8', CacheControl='no-cache')
        print(f"Uploaded {INDEX_KEY} to {bucket}")


if __name__ == "__main__":
    main()
