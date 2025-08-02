def pytest_addoption(parser):
    parser.addoption(
        "--debug-scheduler",
        action="store_true",
        default=False,
        help="Enable debug output for the scheduler related tests",
    )
