class DiagnoseModel:
    def __init__(self, checkpoint, config):
        self.checkpoint = checkpoint
        self.config = config

    def compare_virtual_with_real_trajectories(self, obs, game, horizon):
        # Minimal stub for this demo
        print("diagnose_model stub: comparison skipped")

    def close_all(self):
        pass
