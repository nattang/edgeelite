# QNN SDK Environment Setup Script for Qualcomm Snapdragon X-Elite
# EdgeElite AI Assistant - Qualcomm HaQathon

Write-Host "🚀 Setting up QNN SDK Environment Variables for Snapdragon X-Elite" -ForegroundColor Green
Write-Host "=" * 80

# QNN SDK Path
$QNN_SDK_ROOT = "C:\Users\qc_de\AppData\Roaming\Python\Python313\site-packages\onnxruntime\capi"

# Set environment variables permanently
Write-Host "📝 Setting QNN_SDK_ROOT environment variable..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("QNN_SDK_ROOT", $QNN_SDK_ROOT, "User")

# Add QNN SDK to PATH
Write-Host "📝 Adding QNN SDK to PATH..." -ForegroundColor Yellow
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$QNN_SDK_ROOT*") {
    $newPath = "$currentPath;$QNN_SDK_ROOT"
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Host "✅ Added QNN SDK to PATH" -ForegroundColor Green
} else {
    Write-Host "✅ QNN SDK already in PATH" -ForegroundColor Green
}

# Set additional QNN environment variables
Write-Host "📝 Setting additional QNN environment variables..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("QNN_LOG_LEVEL", "INFO", "User")
[Environment]::SetEnvironmentVariable("QNN_ENABLE_PROFILING", "1", "User")

# Verify the setup
Write-Host "🔍 Verifying QNN SDK setup..." -ForegroundColor Yellow
$env:QNN_SDK_ROOT = $QNN_SDK_ROOT
$env:PATH += ";$QNN_SDK_ROOT"

# Test QNN availability
try {
    $testResult = python -c "import onnxruntime as ort; print('QNN available:', 'QNNExecutionProvider' in ort.get_available_providers())" 2>$null
    if ($testResult -like "*True*") {
        Write-Host "✅ QNN SDK is working correctly!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ QNN SDK test failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Could not test QNN SDK" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 QNN SDK Environment Setup Complete!" -ForegroundColor Green
Write-Host "Environment variables set:" -ForegroundColor Cyan
Write-Host "  - QNN_SDK_ROOT: $QNN_SDK_ROOT" -ForegroundColor White
Write-Host "  - QNN_LOG_LEVEL: INFO" -ForegroundColor White
Write-Host "  - QNN_ENABLE_PROFILING: 1" -ForegroundColor White
Write-Host ""
Write-Host "💡 Note: You may need to restart your terminal or applications" -ForegroundColor Yellow
Write-Host "   for the environment variables to take effect." -ForegroundColor Yellow
Write-Host ""
Write-Host "🚀 Ready for Llama3-TAIDE-LX-8B-Chat-Alpha1 NPU acceleration!" -ForegroundColor Green 