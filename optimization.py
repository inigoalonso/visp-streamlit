import numpy as np
from pymoo.core.problem import ElementwiseProblem
from volume import TotalMechanicalVolumeOfHUD, MirrorFullHeight
from surplus_value import SurplusValue
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=7,
                        n_obj=2,
                        n_constr=1,
                        xl=np.array([5,2,10000,500,70,30,15]),
                        xu=np.array([15,6,30000,1500,210,90,45]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = TotalMechanicalVolumeOfHUD({
                "FullHorizontalFOV" : x[0],
                "FullVerticalFOV" : x[1],
                "VirtualImageDistance" : x[2],
                "EyeboxToMirror1" : x[3],
                "EyeboxFullWidth" : x[4],
                "EyeboxFullHeight" : x[5],
                "Mirror1ObliquityAngle" : x[6],
                "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH" : 70,
                "MechanicalVolumeIncrease" : 40,
                "M1M2OverlapFraction" : 0,
                "PGUVolumeEstimate" : 0.5})
        # f2 has to be negative to maximize the value
        f2 = - SurplusValue(
            x[0],
            x[1],
            MirrorFullHeight({
                "FullHorizontalFOV" : x[0],
                "FullVerticalFOV" : x[1],
                "VirtualImageDistance" : x[2],
                "EyeboxToMirror1" : x[3],
                "EyeboxFullWidth" : x[4],
                "EyeboxFullHeight" : x[5],
                "Mirror1ObliquityAngle" : x[6],
                "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH" : 70,
                "MechanicalVolumeIncrease" : 40,
                "M1M2OverlapFraction" : 0,
                "PGUVolumeEstimate" : 0.5}),
            TotalMechanicalVolumeOfHUD({
                "FullHorizontalFOV" : x[0],
                "FullVerticalFOV" : x[1],
                "VirtualImageDistance" : x[2],
                "EyeboxToMirror1" : x[3],
                "EyeboxFullWidth" : x[4],
                "EyeboxFullHeight" : x[5],
                "Mirror1ObliquityAngle" : x[6],
                "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH" : 70,
                "MechanicalVolumeIncrease" : 40,
                "M1M2OverlapFraction" : 0,
                "PGUVolumeEstimate" : 0.5}),
        )

        # g1 has to be negative to make the volume constraint >= 0
        g1 = - TotalMechanicalVolumeOfHUD({
                "FullHorizontalFOV" : x[0],
                "FullVerticalFOV" : x[1],
                "VirtualImageDistance" : x[2],
                "EyeboxToMirror1" : x[3],
                "EyeboxFullWidth" : x[4],
                "EyeboxFullHeight" : x[5],
                "Mirror1ObliquityAngle" : x[6],
                "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH" : 70,
                "MechanicalVolumeIncrease" : 40,
                "M1M2OverlapFraction" : 0,
                "PGUVolumeEstimate" : 0.5})
        # g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1]

def solve(pop_size=100, n_offsprings=20, n_gen=40):

    # Implement the problem
    problem = MyProblem()

    # Create the algorithm
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        n_offsprings=N_OFFSPRINGS,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    # Termination condition
    termination = get_termination("n_gen", N_GEN)

    # Optimize the problem using NSGA-II
    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)

    X = res.X
    F = res.F

    return res

if __name__ == "__main__":
    print("File one executed when ran directly")
    print(solve(pop_size=100, n_offsprings=20, n_gen=40))
else:
    print("File one executed when imported")
