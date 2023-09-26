#%%
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy
import figureSupportModule as fsm
from string import ascii_lowercase as alph
from matplotlib.image import imread
import scipy.cluster.hierarchy as sch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import radialtree as rt
import sys
sys.path.insert(0, "../topDown")
from referenceMaker import (
    getDefaultReferencesSubdict,
    elaborateDistancesFronReferences,
    getDefaultReferences,
    referenceDendroMaker,
    renamer,
)
#%%
figsize = numpy.array([5, 5]) * 3
NPS = ["to"]
refs = {k: getDefaultReferencesSubdict(k, "../../topDown/References.hdf5") for k in NPS}
refs["icodhto"] = getDefaultReferences("../../topDown/References.hdf5")
cut_at = 0.08
bias = dict(dh=12 + 13, to=12, ico=0)
zoom = 0.01
zoomScissors = zoom * 2
labelPad = 10
#%%
k = "icodhto"
labels = [renamer[k]() for k in refs[k].names]
reorderedColors = [fsm.topDownColorMapHex[i] for i in [0, 1, 2, 4, 5, 3, 6, 7, 9, 8]]
sch.set_link_color_palette(reorderedColors)
wholeDistances = elaborateDistancesFronReferences(refs[k])
Z2 = sch.dendrogram(
    sch.linkage(wholeDistances, method="complete"),
    no_plot=True,
    labels=labels,
    color_threshold=cut_at,
    above_threshold_color="#999",
)
fig, axes = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
linewidht = 3
xmax = numpy.amax(Z2["icoord"])
xmin = numpy.amin(Z2["icoord"])
ymax = numpy.amax(Z2["dcoord"])
ymin = numpy.amin(Z2["dcoord"])
# ax.set_thetalim(np.amin(Z2["icoord"])*2*np.pi,np.amax(Z2["icoord"]))
axes.set_rlim(ymax * 1.01, ymin)
# ax.plot([180, 180], [10, 20])
axes.set_rticks([])
axes.set_xticks(
    [(5 + i * 9.92) / xmax * 2 * numpy.pi for i in range(len(Z2["ivl"]))],
    # [(5 + 46 * 10) / xmax * 2 * numpy.pi for i in range(len(Z2["ivl"]))],
    labels=Z2["ivl"],
)
axes.grid(None)
for icoord, dcoord, c in sorted(zip(Z2["icoord"], Z2["dcoord"], Z2["color_list"])):
    # x, y = Z2['icoord'][0], Z2['dcoord'][0]
    axes.plot(numpy.array(icoord[:2]) / xmax * 2 * numpy.pi, dcoord[:2], c=c)
    axes.plot(
        numpy.linspace(icoord[1], icoord[2], 10) / xmax * 2 * numpy.pi,
        10 * [dcoord[1]],
        c=c,
    )
    axes.plot(numpy.array(icoord[2:]) / xmax * 2 * numpy.pi, dcoord[2:], c=c)
# axes.spines["bottom"].set_position("zero")
axes.spines['polar'].set_visible(False)
axes.tick_params(axis="x", which="major", pad=labelPad)
for i, l in enumerate(Z2["leaves"]):
    img = OffsetImage(imread(f"./topDownFull/topDownFull{l:04}.png"), zoom=zoom)
    n = -0.01
    offset = -0.6
    ab = AnnotationBbox(
        img,
        (5 + i * 10, 0),
        xybox=(5 + i * 10, n + offset) / xmax * 2 * numpy.pi,
        frameon=False,
        xycoords="data",
        # box_alignment=(0.5, 0.5),
    )
    axes.add_artist(ab)
topDownLabels = [
    ("b", 8),  # 0
    ("ss", 7),  # 1
    ("ss'", 3),  # 2
    ("c'", 3),  # 3
    ("s", 8),  # 4
    ("c", 2),  # 5
    ("e", 3),  # 6
    ("e'", 7),  # 7
    ("v", 3),  # 8
    ("v'", 3),  # 9
]
scaledict = {8: 1.75, 7: 1.75, 3: 1.7, 2: 1.6}
pos = 0
myheight = 0.28
axes.set_ylim(bottom=myheight)
axes.set_yticks([])
fig.savefig(f"figureOnlyDendro.png", bbox_inches="tight", pad_inches=0, dpi=300)

# %%
