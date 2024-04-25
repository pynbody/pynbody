import matplotlib.patches
import matplotlib.quiver
import matplotlib.transforms


def _val_or_rc(val, rc_key):
    """Return val if it is not None, otherwise return the value of rc_key in matplotlib.rcParams.

    This is available in some versions of matplotlib, but was added in mid-2023, so we should support
    older versions for now"""

    return val if val is not None else matplotlib.rcParams[rc_key]

class PynbodyQuiverKey(matplotlib.quiver.QuiverKey):
    """An improved version of matplotlib's QuiverKey, allowing a background color to be specified."""

    def __init__(self, *args, **kwargs):
        """An improved quiver key implementation.

        In addition to the arguments of matplotlib.quiver.QuiverKey, additional parameters are below.

        Parameters
        ----------
        boxfacecolor : str or None
            The background color of the key. If None, the default legend face color is used.
        boxedgecolor : str or None
            The edge color of the key. If None, the default legend edge color is used.
        fancybox : bool or None
            If True, the box is drawn with a fancy box style. If None, the default legend fancybox style is used.
            If False, a square-cornered box is drawn.
        *args:
            Additional arguments for matplotlib.quiver.QuiverKey
        **kwargs:
            Additional keyword arguments for matplotlib.quiver.QuiverKey

        """
        self.boxfacecolor = _val_or_rc(kwargs.pop('boxfacecolor', None),
                                                     'legend.facecolor')
        self.boxedgecolor = _val_or_rc(kwargs.pop('boxedgecolor', None),
                                                  'legend.edgecolor')

        if self.boxfacecolor == 'inherit':
            self.boxfacecolor = matplotlib.rcParams['axes.facecolor']

        if self.boxedgecolor == 'inherit':
            self.boxedgecolor = matplotlib.rcParams['axes.edgecolor']

        self.fancybox = _val_or_rc(kwargs.pop("fancybox", None),
                                              'legend.fancybox')

        super().__init__(*args, **kwargs)


    def draw(self, renderer):
        super()._init()

        # the following duplication of bits of super(renderer).draw is necessary to get the
        # text bbox in the right place. Alternative is to actually call super(renderer).draw,
        # but then we end up having to draw twice so that the contents is above the background.
        pos = self.get_transform().transform((self.X, self.Y))
        self.text.set_position(pos + self._text_shift())

        if self.boxfacecolor is not None:
            figure_inverse_trans = self.figure.transFigure.inverted()

            # first find the bbox of the text, in figure coords
            bbox = self.text.get_window_extent(renderer)
            bbox = bbox.transformed(figure_inverse_trans)


            # now find a bbox for the arrow, which is a subtle/annoying thing because the offsets and vertices
            # are transformed differently
            arrow_offsets = self.get_transform().transform(self.vector.get_offsets())
            arrow_vertices =  self.Q.get_transform().transform(self.verts[0])

            arrow_vertices_offset = arrow_offsets + arrow_vertices

            x0y0 = arrow_vertices_offset.min(axis=0)
            x1y1 = arrow_vertices_offset.max(axis=0)

            x0y0 = figure_inverse_trans.transform(x0y0)
            x1y1 = figure_inverse_trans.transform(x1y1)

            # expand bbox to include all arrow_vertices:
            bbox = matplotlib.transforms.Bbox.union([bbox, matplotlib.transforms.Bbox([x0y0, x1y1])])

            # and, at last, we know the coordinates of the background that we need!

            boxstyle = ("round,pad=0.02,rounding_size=0.02" if self.fancybox
                        else "square,pad=0.02")

            background = matplotlib.patches.FancyBboxPatch(
                bbox.min, bbox.width, bbox.height, boxstyle=boxstyle,
                fc=self.boxfacecolor, ec=self.boxedgecolor,
                transform=self.figure.transFigure)

            background.draw(renderer)
        super().draw(renderer)


def _test_quiverkey(scale=10.0, labelpos='E'):
    """A simple test for the PynbodyQuiverKey class."""
    import matplotlib.pyplot as p
    import numpy as np
    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
    U = np.cos(X)
    V = np.sin(Y)
    fig, ax = p.subplots()
    q = ax.quiver(X, Y, U, V)

    qk = PynbodyQuiverKey(q, .5, .95, scale, "Quiver key", labelpos=labelpos,
                          boxfacecolor='w', boxedgecolor='k', fancybox=True)
    ax.add_artist(qk)
    qk.set_zorder(5)
