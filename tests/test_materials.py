import topoheat.multimaterial as multimaterial

def test_multimaterial_gradients():
    er, em, *_ = multimaterial.gradient_check()
    assert er < 1e-4 and em < 1e-4
