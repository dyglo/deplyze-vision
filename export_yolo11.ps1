$ErrorActionPreference = "Stop"

# Use the virtual environment Python
$venvPython = "d:\models_env\Scripts\python.exe"

# Define target output directory
$targetDir = "d:\projects\deplyze-vision\frontend\public\models"
if (!(Test-Path $targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}

$models = @(
    @{ name="yolo11n"; exportName="yolo11n_web_model" },
    @{ name="yolo11n-seg"; exportName="yolo11n-seg_web_model" },
    @{ name="yolo11n-pose"; exportName="yolo11n-pose_web_model" },
    @{ name="yolo11n-obb"; exportName="yolo11n-obb_web_model" }
)

foreach ($model in $models) {
    Write-Host "Exporting $($model.name)..."

    # Run the export command
    # ultralytics export saves the output as {model_name}_web_model in the current dir
    & $venvPython -c "from ultralytics import YOLO; YOLO('$($model.name).pt').export(format='tfjs')"

    $sourceModelDir = ".\$($model.exportName)"
    $destModelDir = Join-Path $targetDir $model.exportName

    if (Test-Path $sourceModelDir) {
        if (Test-Path $destModelDir) {
            Remove-Item -Recurse -Force $destModelDir
        }
        Move-Item $sourceModelDir $destModelDir
        Write-Host "Successfully moved $($model.exportName) to public/models/"
    } else {
        Write-Error "Failed to find generated model directory: $sourceModelDir"
    }
}

Write-Host "All YOLO11 exports complete!"
