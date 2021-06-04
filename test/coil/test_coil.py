from coilpy import Coil
import numpy as np

# read
ellipse = Coil.read_makegrid("ellipse.coils")
assert ellipse.num == 16, "Coil number is read incorrectly!"
assert len(ellipse.data[0].x) == 129, "Segment number is read incorrectly!"
assert ellipse.data[15].I == -1e6, "Coil current is read incorrectly!"
assert ellipse.data[10].group == 3, "Coil group is read incorrectly!"

# plot
ellipse.plot(irange=range(0, 16, 4))
ellipse.plot(irange=range(0, 16, 4), enginer="plotly", plot2d=True)

# save as VTK files
# lines
ellipse.toVTK("ellipse")
# finite-build
label = []
for icoil in list(ellipse):
    zsign = icoil.z[:-1] > 0
    label += zsign.astype(int).tolist()
ellipse.toVTK(
    "ellipse.vtk", line=False, width=0.05, height=0.05, cell_data={"z_sign": [label]}
)

# calculate B field
b = np.array([-5.85704462e-04, 2.94453517e-03, -1.63013362e-18])
assert np.allclose(ellipse.data[0].bfield([0, 0, 0]), b)

# misc
ellipse.data[1].interpolate()
ellipse.data[1].magnify(ratio=2.0)

# save
ellipse.save_makegrid("test.coils")
