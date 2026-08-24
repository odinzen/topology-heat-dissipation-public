"""Run every gradient check and the forward benchmarks in one pass."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import topoheat.topopt_thermoelastic as one
import topoheat.topopt_twoway as two
import topoheat.projected_to as proj
import topoheat.topopt3d as d3
print("one way 2D     :", one.gradient_check()[0])
print("two way 2D     :", two.gradient_check()[0])
print("projected 2D   :", proj.gradient_check())
print("three dimensional:", d3.gradient_check())
