import topoheat.topopt_thermoelastic as topopt_thermoelastic, topoheat.topopt_twoway as topopt_twoway, topoheat.projected_to as projected_to, topoheat.topopt3d as topopt3d, topoheat.stress_to as stress_to

def test_oneway_gradient():
    assert topopt_thermoelastic.gradient_check()[0] < 1e-4

def test_twoway_gradient():
    assert topopt_twoway.gradient_check()[0] < 1e-4

def test_projected_gradient():
    assert projected_to.gradient_check() < 1e-4

def test_3d_gradient():
    assert topopt3d.gradient_check() < 1e-4

def test_stress_gradient():
    assert stress_to.gradient_check() < 1e-4
