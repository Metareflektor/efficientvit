param(
    [switch]$All,
    [switch]$Onnx,
    [switch]$TensorRT,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ModelNames
)

function Show-Usage {
    Write-Host "Usage: $($MyInvocation.MyCommand.Name) [-All | -Onnx | -TensorRT] [model_name(s)]"
    Write-Host "Options:"
    Write-Host "  -All           Generate all models (ONNX models and TensorRT engines)"
    Write-Host "  -Onnx          Generate only ONNX models"
    Write-Host "  -TensorRT      Generate only TensorRT engines"
    Write-Host "Model Names:          l0, l1, l2, xl0, xl1"
    exit 1
}

if ($args[0] -eq "-h" -or $args[0] -eq "--help") {
    Show-Usage
}

$valid_models = @("l0", "l1", "l2", "xl0", "xl1")

if (-not $All -and -not $Onnx -and -not $TensorRT) {
    Write-Host "Error: You must specify one of -All, -Onnx, or -TensorRT options."
    Show-Usage
}

if ($All -and ($Onnx -or $TensorRT)) {
    Write-Host "Error: Options -All, -Onnx, and -TensorRT cannot be used together."
    Show-Usage
}

if ($ModelNames.Count -eq 0) {
    Write-Host "Error: At least one model type (l0, l1, l2, xl0, or xl1) must be specified."
    Show-Usage
}

foreach ($model_name in $ModelNames) {
    if ($valid_models -notcontains $model_name) {
        Write-Host "Error: Invalid model name '$model_name'. Please choose from: $($valid_models -join ', ')"
        Show-Usage
    }
}

function Generate-OnnxModels {
    param($model_name)

    Write-Host "`nCreating $model_name ONNX encoder"
    python deployment/sam/onnx/export_encoder.py `
        --model $model_name `
        --weight_url assets/checkpoints/sam/$model_name.pt `
        --output assets/export_models/sam/onnx/${model_name}_encoder.onnx

    Write-Host "`nCreating $model_name ONNX decoder"
    python deployment/sam/onnx/export_decoder.py `
        --model $model_name `
        --weight_url assets/checkpoints/sam/$model_name.pt `
        --output assets/export_models/sam/onnx/${model_name}_decoder.onnx `
        --return-single-mask
}

function Get-SideLen {
    param($model_name)

    switch -regex ($model_name) {
        '^(l0|l1|l2)$' { return "512" }
        '^(xl0|xl1)$' { return "1024" }
    }
}

function Generate-TensorRTEngines {
    param($model_name)

    $side_len = Get-SideLen $model_name

    Write-Host "`nCreating $model_name TensorRT encoder with side length $side_len"
    trtexec --onnx=assets/export_models/sam/onnx/${model_name}_encoder.onnx `
        --minShapes=input_image:1x3x${side_len}x${side_len} `
        --optShapes=input_image:1x3x${side_len}x${side_len} `
        --maxShapes=input_image:4x3x${side_len}x${side_len} `
        --saveEngine=assets/export_models/sam/tensorrt/${model_name}_encoder.engine

    Write-Host "`nCreating $model_name TensorRT point decoder"
    trtexec --onnx=assets/export_models/sam/onnx/${model_name}_decoder.onnx `
        --minShapes=point_coords:1x1x2,point_labels:1x1 `
        --optShapes=point_coords:1x16x2,point_labels:1x16 `
        --maxShapes=point_coords:1x16x2,point_labels:1x16 `
        --fp16 `
        --saveEngine=assets/export_models/sam/tensorrt/${model_name}_point_decoder.engine

    Write-Host "`nCreating $model_name TensorRT box decoder"
    trtexec --onnx=assets/export_models/sam/onnx/${model_name}_decoder.onnx `
        --minShapes=point_coords:1x1x2,point_labels:1x1 `
        --optShapes=point_coords:16x2x2,point_labels:16x2 `
        --maxShapes=point_coords:16x2x2,point_labels:16x2 `
        --fp16 `
        --saveEngine=assets/export_models/sam/tensorrt/${model_name}_box_decoder.engine

    Write-Host "`nCreating $model_name TensorRT full image segmentation decoder"
    trtexec --onnx=assets/export_models/sam/onnx/${model_name}_decoder.onnx `
        --minShapes=point_coords:1x1x2,point_labels:1x1 `
        --optShapes=point_coords:64x1x2,point_labels:64x1 `
        --maxShapes=point_coords:128x1x2,point_labels:128x1 `
        --fp16 `
        --saveEngine=assets/export_models/sam/tensorrt/${model_name}_full_img_decoder.engine
}

if ($All -or $Onnx) {
    New-Item -ItemType Directory -Force -Path assets/export_models/sam/onnx | Out-Null

    foreach ($model_name in $ModelNames) {
        Generate-OnnxModels $model_name
    }
}

if ($All -or $TensorRT) {
    New-Item -ItemType Directory -Force -Path assets/export_models/sam/tensorrt | Out-Null

    foreach ($model_name in $ModelNames) {
        Generate-TensorRTEngines $model_name
    }
}
