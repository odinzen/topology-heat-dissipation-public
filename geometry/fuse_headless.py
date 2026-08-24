# Headless fuse: turn the multi-solid design.step (one box per voxel) into ONE
# clean fused solid and write design_solid.step. Runs without the GUI:
#
#     freecadcmd geometry/fuse_headless.py
#
# A PNG render needs the GUI, so this headless run skips it and reports the fact.
# For the render, run fuse_and_render.FCMacro inside the FreeCAD GUI instead.

import os
import time
import FreeCAD as App
import Part

# Locate files relative to this script so it runs from any shell or OS.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
BASE     = os.path.join(HERE, "truck_step")
STEP_IN  = os.path.join(BASE, "design.step")
STEP_OUT = os.path.join(BASE, "design_solid.step")

t0 = time.time()
shape = Part.Shape()
shape.read(STEP_IN)
solids = list(shape.Solids)
App.Console.PrintMessage("read %d solids\n" % len(solids))

# Pairwise (divide and conquer) fusion is far faster than a sequential loop.
def fuse_all(items):
    level = 0
    while len(items) > 1:
        nxt = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                nxt.append(items[i].fuse(items[i + 1]))
            else:
                nxt.append(items[i])
        items = nxt
        level += 1
        App.Console.PrintMessage("  level %d -> %d shapes\n" % (level, len(items)))
    return items[0]

fused = fuse_all(solids)
fused = fused.removeSplitter()   # merge coplanar faces into clean faces
App.Console.PrintMessage(
    "fused: %d solids -> 1, volume %.1f, %d faces, %.1f s\n"
    % (len(solids), fused.Volume, len(fused.Faces), time.time() - t0)
)
fused.exportStep(STEP_OUT)
App.Console.PrintMessage("wrote %s\n" % STEP_OUT)
