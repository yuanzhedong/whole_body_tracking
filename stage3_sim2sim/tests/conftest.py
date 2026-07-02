def pytest_configure(config):
    config.addinivalue_line("markers", "slow: integration tests that run the real HoloMotion tracker + MuJoCo")
