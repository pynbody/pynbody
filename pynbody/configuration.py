"""Read and expose configuration information for pynbody

The configuration system in pynbody is described in the :ref:`configuration` tutorial.

"""

import logging
import multiprocessing
import os
import warnings


def _get_config_parser_with_defaults():
    # Create config dictionaries which will be required by subpackages
    import configparser
    config_parser = configparser.ConfigParser()
    config_parser.optionxform = str
    config_parser.read(
        os.path.join(os.path.dirname(__file__), "default_config.ini"))
    return config_parser

def _merge_defaults_for_problematic_keys(config_parser):
    """This unfortunate routine is made necessary by issue #261"""
    config_parser_defaults = _get_config_parser_with_defaults()
    merge = (('irreducible-units','names'),)

    for merge_i in merge:
        opt = config_parser.get(*merge_i)
        default_opt = config_parser_defaults.get(*merge_i)

        items = list(map(str.strip,opt.split(",")))
        default_items = list(map(str.strip,default_opt.split(",")))

        for checking_item in default_items:
            if checking_item not in items:
                warnings.warn("Pynbody spotted a potential problem with your .pynbodyrc or config.ini. Overriding it by adding %r to config section %r item %r"%(checking_item, merge_i[0], merge_i[1]))
                opt+=", "+checking_item
                config_parser.set(merge_i[0],merge_i[1],opt)

def _add_overrides_to_config_parser(config_parser):
    config_parser.read(os.path.join(os.path.dirname(__file__), "config.ini"))
    config_parser.read(os.path.expanduser("~/.pynbodyrc"))
    config_parser.read("config.ini")
    _merge_defaults_for_problematic_keys(config_parser)

def _get_basic_config_from_parser(config_parser):

    config = {'centering-scheme': config_parser.get('general', 'centering-scheme')}

    config['snap-class-priority'] = list(map(str.strip,
                                        config_parser.get('general', 'snap-class-priority').split(",")))
    config['halo-class-priority'] = list(map(str.strip,
                                        config_parser.get('general', 'halo-class-priority').split(",")))


    config['default-cosmology'] = {}
    for k in config_parser.options('default-cosmology'):
        config[
            'default-cosmology'][k] = float(config_parser.get('default-cosmology', k))

    config['sph'] = {}
    for k in config_parser.options('sph'):
        try:
            config['sph'][k] = int(config_parser.get('sph', k))
        except ValueError:
            pass
    config['sph']['kernel'] = config_parser.get('sph', 'kernel')

    config['threading'] = config_parser.get('general', 'threading')
    config['number_of_threads'] = int(
        config_parser.get('general', 'number_of_threads'))

    if config['number_of_threads']<0:
        config['number_of_threads']=multiprocessing.cpu_count()

    config['gravity_calculation_mode'] = config_parser.get(
        'general', 'gravity_calculation_mode')
    config['disk-fit-function'] = config_parser.get('general', 'disk-fit-function')

    config['image-default-resolution'] = int(config_parser.get('general', 'image-default-resolution'))
    config['image-default-nside'] = int(config_parser.get('general', 'image-default-nside'))
    return config

def _setup_logger(config):
    logger = logging.getLogger('pynbody')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s : %(message)s')
    for existing_handler in list(logger.handlers):
        logger.removeHandler(existing_handler)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if config_parser.getboolean('general','verbose'):
        set_logging_level(logging.INFO)
        logger.info("Verbose mode is on")
    else:
        set_logging_level(logging.WARNING)

def set_logging_level(level = logging.INFO):
    """Set the logging level for pynbody, in terms of the standard Python logging module levels.

    Set to logging.INFO for more verbose output, or logging.WARNING for less."""
    logger = logging.getLogger('pynbody')
    logger.setLevel(level)


config_parser = _get_config_parser_with_defaults()
_add_overrides_to_config_parser(config_parser)
config = _get_basic_config_from_parser(config_parser)
logger = _setup_logger(config)
