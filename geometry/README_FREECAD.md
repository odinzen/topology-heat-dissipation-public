# Cleaning the STEP in FreeCAD (optional)

design.step from the Truck exporter is the optimized design as 1604 separate
B-Rep box solids. It opens in FreeCAD as is. To turn it into one clean fused
solid and a render, use the macro fuse_and_render.FCMacro.

## Steps
1. Open FreeCAD.
2. Macro menu, then Macros..., create a macro, paste the contents of
   fuse_and_render.FCMacro, and Execute. Or open the Python console
   (View, Panels, Python console) and paste it there.
3. Edit the two paths at the top if your files are not under ~/te.

It writes design_solid.step, a single fused solid, next to design.step, and
saves design_render.png in ~/te.

## Notes
Fusing 1604 solids takes a minute or two and some memory; the pairwise method in
the macro keeps it tractable. FreeCAD itself is LGPL, fine to use as an
application; it is not bundled or redistributed by the engine, so the permissive
licensing of the shipped stack is unaffected. If fusion is slow on your machine,
the multi solid design.step is already valid CAD and opens directly.
