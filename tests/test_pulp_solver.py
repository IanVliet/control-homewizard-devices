import pulp as pl


def test_list_solvers_contains_SCIP():
    """
    Test that the list of solvers contains SCIP.
    """
    solver_list = pl.listSolvers(onlyAvailable=True)
    assert "SCIP_PY" in solver_list, (
        "SCIP solver should be available in the list of solvers."
    )


def test_list_solvers_contains_CBC():
    """
    Test that the list of solvers contains CBC.
    """
    solver_list = pl.listSolvers(onlyAvailable=True)
    assert "PULP_CBC_CMD" in solver_list, (
        "The PULP_CBC_CMD solver should be available in the list of solvers."
    )
