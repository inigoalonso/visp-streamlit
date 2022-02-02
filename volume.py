import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# HUD Volume Estimation

## Inputs
inputs = {
    "FullHorizontalFOV" : 10,
    "FullVerticalFOV" : 4,
    "VirtualImageDistance" : 15000,
    "EyeboxToMirror1" : 1000,
    "EyeboxFullWidth" : 140,
    "EyeboxFullHeight" : 60,
    "Mirror1ObliquityAngle" : 30,
    "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH" : 70,
    "MechanicalVolumeIncrease" : 40,
    "M1M2OverlapFraction" : 0,
    "PGUVolumeEstimate" : 0.5
}

def ScaledDiffuserScreenWidth(inputs):
    ScalingFactor = inputs["FullHorizontalFOV"]/10
    return inputs["HUD_SCREEN_10x5_FOV_BASELINE_WIDTH"] * ScalingFactor

HUDDiffuserScreenImageWidth = ScaledDiffuserScreenWidth

## Calculations
### Mirror Size Calculations
#### Mirror 1 Horizontal Width Calculation
def RightField(inputs):
    points = {
        "RightFieldFieldPoint1XH" : -inputs["VirtualImageDistance"],
        "RightFieldFieldPoint1YH" : np.tan(np.radians(inputs["FullHorizontalFOV"]/2))*inputs["VirtualImageDistance"],
        "RightFieldWideSideEyebox1XH" : 0,
        "RightFieldWideSideEyebox1YH" : inputs["EyeboxFullWidth"]/2,
        "RightFieldFieldPoint2XH" : -inputs["VirtualImageDistance"],
        "RightFieldFieldPoint2YH" : np.tan(np.radians(inputs["FullHorizontalFOV"]/2))*inputs["VirtualImageDistance"],
        "RightFieldWideSideEyebox2XH" : 0,
        "RightFieldWideSideEyebox2YH" : 0,
        "RightFieldFieldPoint3XH" : -inputs["VirtualImageDistance"],
        "RightFieldFieldPoint3YH" : np.tan(np.radians(inputs["FullHorizontalFOV"]/2))*inputs["VirtualImageDistance"],
        "RightFieldWideSideEyebox3XH" : 0,
        "RightFieldWideSideEyebox3YH" : -inputs["EyeboxFullWidth"]/2
    }
    return points

def HorizontalCenterField(inputs):
    points = {
        "CenterFieldFieldPoint1XH" : -inputs["VirtualImageDistance"],
        "CenterFieldFieldPoint1YH" : 0, #=TAN(RADIANS(B1/2))*C3 where B1 and C3 are empty
        "CenterFieldWideSideEyebox1XH" : 0,
        "CenterFieldWideSideEyebox1YH" : inputs["EyeboxFullWidth"]/2,
        "CenterFieldFieldPoint2XH" : -inputs["VirtualImageDistance"],
        "CenterFieldFieldPoint2YH" : 0,
        "CenterFieldWideSideEyebox2XH" : 0,
        "CenterFieldWideSideEyebox2YH" : 0,
        "CenterFieldFieldPoint3XH" : -inputs["VirtualImageDistance"],
        "CenterFieldFieldPoint3YH" : 0,
        "CenterFieldWideSideEyebox3XH" : 0,
        "CenterFieldWideSideEyebox3YH" : -inputs["EyeboxFullWidth"]/2
    }
    return points

def LeftField(inputs):
    points = {
        "LeftFieldFieldPoint1XH" : -inputs["VirtualImageDistance"],
        "LeftFieldFieldPoint1YH" : -RightField(inputs)["RightFieldFieldPoint1YH"],
        "LeftFieldWideSideEyebox1XH" : 0,
        "LeftFieldWideSideEyebox1YH" : inputs["EyeboxFullWidth"]/2,
        "LeftFieldFieldPoint2XH" : -inputs["VirtualImageDistance"],
        "LeftFieldFieldPoint2YH" : -RightField(inputs)["RightFieldFieldPoint1YH"],
        "LeftFieldWideSideEyebox2XH" : 0,
        "LeftFieldWideSideEyebox2YH" : 0,
        "LeftFieldFieldPoint3XH" : -inputs["VirtualImageDistance"],
        "LeftFieldFieldPoint3YH" : -RightField(inputs)["RightFieldFieldPoint1YH"],
        "LeftFieldWideSideEyebox3XH" : 0,
        "LeftFieldWideSideEyebox3YH" : -inputs["EyeboxFullWidth"]/2
    }
    return points

def VirtualImageHorizontal(inputs):
    points = {
        "VirtualImageTopPointXH" : RightField(inputs)["RightFieldFieldPoint1XH"],
        "VirtualImageTopPointYH" : RightField(inputs)["RightFieldFieldPoint1YH"],
        "VirtualImageBottomPointXH" : RightField(inputs)["RightFieldFieldPoint1XH"],
        "VirtualImageBottomPointYH" : -RightField(inputs)["RightFieldFieldPoint1YH"]
    }
    return points

##### Mirror 1
def RaySlopeH_m(inputs):
    return (np.tan(np.radians(inputs["FullHorizontalFOV"]/2))*inputs["VirtualImageDistance"]-inputs["EyeboxFullWidth"]/2)/(-inputs["VirtualImageDistance"])
    
def InterceptH(inputs):
    return inputs["EyeboxFullWidth"]/2

def Mirror1Horizontal(inputs):
    points = {
        "Mirror1TopPointXH" : -inputs["EyeboxToMirror1"],
        "Mirror1TopPointYH" : RaySlopeH_m(inputs) * (-inputs["EyeboxToMirror1"]) + InterceptH(inputs),
        "Mirror1BottomPointXH" : -inputs["EyeboxToMirror1"],
        "Mirror1BottomPointYH" : -RaySlopeH_m(inputs) * (-inputs["EyeboxToMirror1"]) + InterceptH(inputs)
    }
    return points

sns.set_theme(style="ticks")

dataHorizontal = [
    {'group': 'Right Field', 'x': RightField(inputs)["RightFieldFieldPoint1XH"], 'y': RightField(inputs)["RightFieldFieldPoint1YH"]},
    {'group': 'Right Field', 'x': RightField(inputs)["RightFieldWideSideEyebox1XH"], 'y': RightField(inputs)["RightFieldWideSideEyebox1YH"]},
    {'group': 'Right Field', 'x': RightField(inputs)["RightFieldFieldPoint2XH"], 'y': RightField(inputs)["RightFieldFieldPoint2YH"]},
    {'group': 'Right Field', 'x': RightField(inputs)["RightFieldWideSideEyebox2XH"], 'y': RightField(inputs)["RightFieldWideSideEyebox2YH"]},
    {'group': 'Right Field', 'x': RightField(inputs)["RightFieldFieldPoint3XH"], 'y': RightField(inputs)["RightFieldFieldPoint3YH"]},
    {'group': 'Right Field', 'x': RightField(inputs)["RightFieldWideSideEyebox3XH"], 'y': RightField(inputs)["RightFieldWideSideEyebox3YH"]},
    {'group': 'Center Field', 'x': HorizontalCenterField(inputs)["CenterFieldFieldPoint1XH"], 'y': HorizontalCenterField(inputs)["CenterFieldFieldPoint1YH"]},
    {'group': 'Center Field', 'x': HorizontalCenterField(inputs)["CenterFieldWideSideEyebox1XH"], 'y': HorizontalCenterField(inputs)["CenterFieldWideSideEyebox1YH"]},
    {'group': 'Center Field', 'x': HorizontalCenterField(inputs)["CenterFieldFieldPoint2XH"], 'y': HorizontalCenterField(inputs)["CenterFieldFieldPoint2YH"]},
    {'group': 'Center Field', 'x': HorizontalCenterField(inputs)["CenterFieldWideSideEyebox2XH"], 'y': HorizontalCenterField(inputs)["CenterFieldWideSideEyebox2YH"]},
    {'group': 'Center Field', 'x': HorizontalCenterField(inputs)["CenterFieldFieldPoint3XH"], 'y': HorizontalCenterField(inputs)["CenterFieldFieldPoint3YH"]},
    {'group': 'Center Field', 'x': HorizontalCenterField(inputs)["CenterFieldWideSideEyebox3XH"], 'y': HorizontalCenterField(inputs)["CenterFieldWideSideEyebox3YH"]},
    {'group': 'Left Field', 'x': LeftField(inputs)["LeftFieldFieldPoint1XH"], 'y': LeftField(inputs)["LeftFieldFieldPoint1YH"]},
    {'group': 'Left Field', 'x': LeftField(inputs)["LeftFieldWideSideEyebox1XH"], 'y': LeftField(inputs)["LeftFieldWideSideEyebox1YH"]},
    {'group': 'Left Field', 'x': LeftField(inputs)["LeftFieldFieldPoint2XH"], 'y': LeftField(inputs)["LeftFieldFieldPoint2YH"]},
    {'group': 'Left Field', 'x': LeftField(inputs)["LeftFieldWideSideEyebox2XH"], 'y': LeftField(inputs)["LeftFieldWideSideEyebox2YH"]},
    {'group': 'Left Field', 'x': LeftField(inputs)["LeftFieldFieldPoint3XH"], 'y': LeftField(inputs)["LeftFieldFieldPoint3YH"]},
    {'group': 'Left Field', 'x': LeftField(inputs)["LeftFieldWideSideEyebox3XH"], 'y': LeftField(inputs)["LeftFieldWideSideEyebox3YH"]},
    {'group': 'Virtual Image', 'x': VirtualImageHorizontal(inputs)["VirtualImageTopPointXH"], 'y': VirtualImageHorizontal(inputs)["VirtualImageTopPointYH"]},
    {'group': 'Virtual Image', 'x': VirtualImageHorizontal(inputs)["VirtualImageBottomPointXH"], 'y': VirtualImageHorizontal(inputs)["VirtualImageBottomPointYH"]},
    {'group': 'Mirror 1', 'x': Mirror1Horizontal(inputs)["Mirror1TopPointXH"], 'y': Mirror1Horizontal(inputs)["Mirror1TopPointYH"]},
    {'group': 'Mirror 1', 'x': Mirror1Horizontal(inputs)["Mirror1BottomPointXH"], 'y': Mirror1Horizontal(inputs)["Mirror1BottomPointYH"]}
    ]
dfHorizontal = pd.DataFrame.from_records(dataHorizontal)

sns.lineplot(
    data=dfHorizontal,
    x="x", y="y", hue="group", style="group",
    markers=True, dashes=False
)

def TopField(inputs):
    points = {
        "TopFieldFieldPoint1XV" : -inputs["VirtualImageDistance"],
        "TopFieldFieldPoint1YV" : np.tan(np.radians(inputs["FullVerticalFOV"]/2))*inputs["VirtualImageDistance"],
        "TopFieldWideSideEyebox1XV" : 0,
        "TopFieldWideSideEyebox1YV" : inputs["EyeboxFullHeight"]/2,
        "TopFieldFieldPoint2XV" : -inputs["VirtualImageDistance"],
        "TopFieldFieldPoint2YV" : np.tan(np.radians(inputs["FullVerticalFOV"]/2))*inputs["VirtualImageDistance"],
        "TopFieldWideSideEyebox2XV" : 0,
        "TopFieldWideSideEyebox2YV" : 0,
        "TopFieldFieldPoint3XV" : -inputs["VirtualImageDistance"],
        "TopFieldFieldPoint3YV" : np.tan(np.radians(inputs["FullVerticalFOV"]/2))*inputs["VirtualImageDistance"],
        "TopFieldWideSideEyebox3XV" : 0,
        "TopFieldWideSideEyebox3YV" : -inputs["EyeboxFullHeight"]/2
    }
    return points

def VerticalCenterField(inputs):
    points = {
        "CenterFieldFieldPoint1XV" : -inputs["VirtualImageDistance"],
        "CenterFieldFieldPoint1YV" : 0, #=TAN(RADIANS(B43/2))*C45 where B43 and C45 are empty
        "CenterFieldWideSideEyebox1XV" : 0,
        "CenterFieldWideSideEyebox1YV" : inputs["EyeboxFullHeight"]/2,
        "CenterFieldFieldPoint2XV" : -inputs["VirtualImageDistance"],
        "CenterFieldFieldPoint2YV" : 0,
        "CenterFieldWideSideEyebox2XV" : 0,
        "CenterFieldWideSideEyebox2YV" : 0,
        "CenterFieldFieldPoint3XV" : -inputs["VirtualImageDistance"],
        "CenterFieldFieldPoint3YV" : 0,
        "CenterFieldWideSideEyebox3XV" : 0,
        "CenterFieldWideSideEyebox3YV" : -inputs["EyeboxFullHeight"]/2
    }
    return points

def BottomField(inputs):
    points = {
        "BottomFieldFieldPoint1XV" : -inputs["VirtualImageDistance"],
        "BottomFieldFieldPoint1YV" : -TopField(inputs)["TopFieldFieldPoint1YV"],
        "BottomFieldWideSideEyebox1XV" : 0,
        "BottomFieldWideSideEyebox1YV" : TopField(inputs)["TopFieldWideSideEyebox1YV"],
        "BottomFieldFieldPoint2XV" : -inputs["VirtualImageDistance"],
        "BottomFieldFieldPoint2YV" : -TopField(inputs)["TopFieldFieldPoint1YV"],
        "BottomFieldWideSideEyebox2XV" : 0,
        "BottomFieldWideSideEyebox2YV" : 0,
        "BottomFieldFieldPoint3XV" : -inputs["VirtualImageDistance"],
        "BottomFieldFieldPoint3YV" : -TopField(inputs)["TopFieldFieldPoint1YV"],
        "BottomFieldWideSideEyebox3XV" : 0,
        "BottomFieldWideSideEyebox3YV" : -TopField(inputs)["TopFieldWideSideEyebox1YV"]
    }
    return points


def VirtualImageVertical(inputs):
    points = {
        "VirtualImageTopPointXV" : TopField(inputs)["TopFieldFieldPoint1XV"],
        "VirtualImageTopPointYV" : TopField(inputs)["TopFieldFieldPoint1YV"],
        "VirtualImageBottomPointXV" : TopField(inputs)["TopFieldFieldPoint1XV"],
        "VirtualImageBottomPointYV" : -TopField(inputs)["TopFieldFieldPoint1YV"]
    }
    return points

##### Mirror 1
def RaySlopeV_m(inputs):
    return (np.tan(np.radians(inputs["FullVerticalFOV"]/2))*inputs["VirtualImageDistance"]-inputs["EyeboxFullHeight"]/2)/(-inputs["VirtualImageDistance"])

def InterceptV(inputs):
    return inputs["EyeboxFullHeight"]/2

def Mirror1Vertical(inputs):
    points = {
        "Mirror1TopPointXV" : -inputs["EyeboxToMirror1"],
        "Mirror1TopPointYV" : RaySlopeV_m(inputs) * (-inputs["EyeboxToMirror1"]) + InterceptV(inputs),
        "Mirror1BottomPointXV" : -inputs["EyeboxToMirror1"],
        "Mirror1BottomPointYV" : -RaySlopeV_m(inputs) * (-inputs["EyeboxToMirror1"]) + InterceptV(inputs)
    }
    return points

sns.set_theme(style="ticks")

dataVertical = [
    {'group': 'Top Field', 'x': TopField(inputs)["TopFieldFieldPoint1XV"], 'y': TopField(inputs)["TopFieldFieldPoint1YV"]},
    {'group': 'Top Field', 'x': TopField(inputs)["TopFieldWideSideEyebox1XV"], 'y': TopField(inputs)["TopFieldWideSideEyebox1YV"]},
    {'group': 'Top Field', 'x': TopField(inputs)["TopFieldFieldPoint2XV"], 'y': TopField(inputs)["TopFieldFieldPoint2YV"]},
    {'group': 'Top Field', 'x': TopField(inputs)["TopFieldWideSideEyebox2XV"], 'y': TopField(inputs)["TopFieldWideSideEyebox2YV"]},
    {'group': 'Top Field', 'x': TopField(inputs)["TopFieldFieldPoint3XV"], 'y': TopField(inputs)["TopFieldFieldPoint3YV"]},
    {'group': 'Top Field', 'x': TopField(inputs)["TopFieldWideSideEyebox3XV"], 'y': TopField(inputs)["TopFieldWideSideEyebox3YV"]},
    {'group': 'Center Field', 'x': VerticalCenterField(inputs)["CenterFieldFieldPoint1XV"], 'y': VerticalCenterField(inputs)["CenterFieldFieldPoint1YV"]},
    {'group': 'Center Field', 'x': VerticalCenterField(inputs)["CenterFieldWideSideEyebox1XV"], 'y': VerticalCenterField(inputs)["CenterFieldWideSideEyebox1YV"]},
    {'group': 'Center Field', 'x': VerticalCenterField(inputs)["CenterFieldFieldPoint2XV"], 'y': VerticalCenterField(inputs)["CenterFieldFieldPoint2YV"]},
    {'group': 'Center Field', 'x': VerticalCenterField(inputs)["CenterFieldWideSideEyebox2XV"], 'y': VerticalCenterField(inputs)["CenterFieldWideSideEyebox2YV"]},
    {'group': 'Center Field', 'x': VerticalCenterField(inputs)["CenterFieldFieldPoint3XV"], 'y': VerticalCenterField(inputs)["CenterFieldFieldPoint3YV"]},
    {'group': 'Center Field', 'x': VerticalCenterField(inputs)["CenterFieldWideSideEyebox3XV"], 'y': VerticalCenterField(inputs)["CenterFieldWideSideEyebox3YV"]},
    {'group': 'Bottom Field', 'x': BottomField(inputs)["BottomFieldFieldPoint1XV"], 'y': BottomField(inputs)["BottomFieldFieldPoint1YV"]},
    {'group': 'Bottom Field', 'x': BottomField(inputs)["BottomFieldWideSideEyebox1XV"], 'y': BottomField(inputs)["BottomFieldWideSideEyebox1YV"]},
    {'group': 'Bottom Field', 'x': BottomField(inputs)["BottomFieldFieldPoint2XV"], 'y': BottomField(inputs)["BottomFieldFieldPoint2YV"]},
    {'group': 'Bottom Field', 'x': BottomField(inputs)["BottomFieldWideSideEyebox2XV"], 'y': BottomField(inputs)["BottomFieldWideSideEyebox2YV"]},
    {'group': 'Bottom Field', 'x': BottomField(inputs)["BottomFieldFieldPoint3XV"], 'y': BottomField(inputs)["BottomFieldFieldPoint3YV"]},
    {'group': 'Bottom Field', 'x': BottomField(inputs)["BottomFieldWideSideEyebox3XV"], 'y': BottomField(inputs)["BottomFieldWideSideEyebox3YV"]},
    {'group': 'Virtual Image', 'x': VirtualImageVertical(inputs)["VirtualImageTopPointXV"], 'y': VirtualImageVertical(inputs)["VirtualImageTopPointYV"]},
    {'group': 'Virtual Image', 'x': VirtualImageVertical(inputs)["VirtualImageBottomPointXV"], 'y': VirtualImageVertical(inputs)["VirtualImageBottomPointYV"]},
    {'group': 'Mirror 1', 'x': Mirror1Vertical(inputs)["Mirror1TopPointXV"], 'y': Mirror1Vertical(inputs)["Mirror1TopPointYV"]},
    {'group': 'Mirror 1', 'x': Mirror1Vertical(inputs)["Mirror1BottomPointXV"], 'y': Mirror1Vertical(inputs)["Mirror1BottomPointYV"]}
    ]
dfVertical = pd.DataFrame.from_records(dataVertical)

sns.lineplot(
    data=dfVertical,
    x="x", y="y", hue="group", style="group",
    markers=True, dashes=False
)

def MirrorFullHeight(inputs):
    #TiltAngleOfM1 = Mirror1ObliquityAngle #degrees
    TopPointX    = ( -InterceptV(inputs) * np.tan(np.radians(inputs["Mirror1ObliquityAngle"])) - inputs["EyeboxToMirror1"]) / (1+np.tan(np.radians(inputs["Mirror1ObliquityAngle"])) * RaySlopeV_m(inputs))
    BottomPointX = (  InterceptV(inputs) * np.tan(np.radians(inputs["Mirror1ObliquityAngle"])) - inputs["EyeboxToMirror1"]) / (1-np.tan(np.radians(inputs["Mirror1ObliquityAngle"])) * RaySlopeV_m(inputs))

    TopPointY    = (  RaySlopeV_m(inputs) * TopPointX    ) + InterceptV(inputs)
    BottomPointY = ( -RaySlopeV_m(inputs) * BottomPointX ) - InterceptV(inputs)

    return np.sqrt(np.power(TopPointX-BottomPointX,2)+np.power(TopPointY-BottomPointY,2))

#### M1 focal length calculation
def M1ToVirtualImage(inputs):
    return inputs["VirtualImageDistance"]-inputs["EyeboxToMirror1"] #(s2) mm

def SceenToM1(inputs):
    ## Screen and Image Calculations		
    VirtualImageWidth = np.tan(np.radians(inputs["FullHorizontalFOV"]/2))*inputs["VirtualImageDistance"]*2 #mm
    HUDImageMagnification = VirtualImageWidth/HUDDiffuserScreenImageWidth(inputs)	#x
    return M1ToVirtualImage(inputs)/HUDImageMagnification #mm

# KEEP!
# Not used in volume estimation
#ScreenSize = HUDDiffuserScreenImageWidth #mm
#OpticalPowerOfM1 = (1/SceenToM1(inputs))-(1/M1ToVirtualImage(inputs)) #1/f = 1/s1+1/s2 (note s2 is negative since it’s a virtual image)
#FocalLengthOfM1 = 1/OpticalPowerOfM1 #(f) mm

### HUD Volume Calculation
def VolumeBetweenScreenAndM1Liters(inputs):
    ## Primary mirror size estimate
    Mirror1HorizontalDiameterEstimate = Mirror1Horizontal(inputs)["Mirror1TopPointYH"]*2 #mm
    Mirror1VerticalDiameterEstimate = MirrorFullHeight(inputs) #mm

    Mirror1Width = Mirror1HorizontalDiameterEstimate #mm
    Mirror1Height = Mirror1VerticalDiameterEstimate #mm
    Mirror1Area = Mirror1Width*Mirror1Height #mm^2
            
    ScreenToM1 = SceenToM1(inputs) #mm
    ScreenWidth = HUDDiffuserScreenImageWidth(inputs) #mm
    ScreenHeight = (inputs["FullVerticalFOV"]/inputs["FullHorizontalFOV"])*HUDDiffuserScreenImageWidth(inputs) #mm
            
    # Find height of pyramid from base		
    HalfDiagonalM1 = 0.5*(np.sqrt(np.power(Mirror1Width,2)+np.power(Mirror1Height,2))) #Y2
    HalfDiagonalScreen = 0.5*(np.sqrt(np.power(ScreenWidth,2)+np.power(ScreenHeight,2))) #Y1
    Slope = (HalfDiagonalScreen-HalfDiagonalM1)/ScreenToM1 #(m)
    #x=0, y=0, so b=0		
    X2 = HalfDiagonalM1/Slope
    X1 = HalfDiagonalScreen/Slope
    ApexFromMirror1 = np.absolute(X2)
    ApexFromScreen = np.absolute(X1)

    # Find Mirror to screen volume
    AreaPyramidFromMirror = (1/3)*Mirror1Width*Mirror1Height*ApexFromMirror1 #mm^3
    AreaPyramidFromScreen = (1/3)*ScreenWidth*ScreenHeight*ApexFromScreen #mm^3
    VolumeBetweenScreenAndM1 = AreaPyramidFromMirror-AreaPyramidFromScreen #mm^3

    return 0.000001*VolumeBetweenScreenAndM1

## Outputs
### HUD Volume Estimate
def TotalMechanicalVolumeOfHUD(inputs):
    ## Primary mirror size estimate
    Mirror1HorizontalDiameterEstimate = Mirror1Horizontal(inputs)["Mirror1TopPointYH"]*2 #mm
    Mirror1VerticalDiameterEstimate = MirrorFullHeight(inputs) #mm

    VolumeFromHUDmirrors = VolumeBetweenScreenAndM1Liters(inputs)*(1-inputs["M1M2OverlapFraction"]/100) #Liters
    #print("Volume from HUD mirrors: {:.2f} liters".format(VolumeFromHUDmirrors))
    VolumeFromMirrorsAndPGU = VolumeFromHUDmirrors+inputs["PGUVolumeEstimate"] #Liters
    #print("Volume from mirrors + PGU: {:.2f} liters".format(VolumeFromMirrorsAndPGU))
    total = VolumeFromMirrorsAndPGU*(1+inputs["MechanicalVolumeIncrease"]/100)
    #print("Total Mechanical Volume of HUD: {:.2f} liters".format(total))
    return total
