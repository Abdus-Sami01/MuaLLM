# Azure setup for the MuaLLM project. Idempotent - safe to re-run.
#
# Provisions:
#   - resource group
#   - storage account + containers (corpus, logits, checkpoints, tokenizer)
#   - Azure ML workspace
#   - low-priority (spot) GPU compute cluster (Standard_NC4as_T4_v3, scale 0..1)
#   - monthly budget with alerts at 50/75/90 %
#
# Prereqs:
#   winget install Microsoft.AzureCLI
#   az login
#   az extension add --name ml --upgrade --yes
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\azure_setup.ps1
#
# Cost expectation on a $200 credit pool (T4 spot ~$0.10-0.15/hr):
#   - logit cache (~22 h)  ~$3
#   - distill train (~20 h) ~$3
#   - ablation 3 variants  ~$9
#   - buffer + storage     ~$12
#   total                  ~$27 spot / ~$96 on-demand
#
$ErrorActionPreference = "Stop"

# ----------------------------- config -----------------------------
$RG          = "mua-llm-rg"
$LOCATION    = "southcentralus"        # T4 spot tends to be available here
$STORAGE     = "muallm" + (Get-Random -Maximum 999999).ToString("D6")
$WORKSPACE   = "mua-llm-ws"
$COMPUTE     = "t4-spot"
$VM_SIZE     = "Standard_NC4as_T4_v3"  # 1x T4 16GB, 4 vCPU
$BUDGET_NAME = "mua-llm-budget"
$BUDGET_USD  = 180                     # below your $200 to leave headroom
$IDLE_SECS   = 600

# ----------------------------- preflight -----------------------------
Write-Host "checking az CLI..."
$null = az --version
if ($LASTEXITCODE -ne 0) { throw "az CLI not found. Install via winget." }

$SUB   = az account show --query id -o tsv
$EMAIL = az account show --query user.name -o tsv
if (-not $SUB)   { throw "not logged in. Run 'az login' first." }
if (-not $EMAIL) { $EMAIL = Read-Host "Enter contact email for budget alerts" }

Write-Host ""
Write-Host "subscription : $SUB"
Write-Host "user         : $EMAIL"
Write-Host "rg           : $RG"
Write-Host "location     : $LOCATION"
Write-Host "storage acct : $STORAGE"
Write-Host "workspace    : $WORKSPACE"
Write-Host "compute      : $COMPUTE ($VM_SIZE, LowPriority, 0..1)"
Write-Host "budget       : `$$BUDGET_USD / month"
Write-Host ""

# ----------------------------- 1/6 resource group -----------------------------
Write-Host "[1/6] resource group"
$rgExists = az group exists --name $RG | ConvertFrom-Json
if (-not $rgExists) {
    az group create --name $RG --location $LOCATION | Out-Null
}
Write-Host "  ok"

# ----------------------------- 2/6 storage -----------------------------
Write-Host "[2/6] storage account + containers"
$existing = az storage account list -g $RG --query "[].name" -o tsv
if ($existing) {
    $STORAGE = ($existing -split "`n")[0]
    Write-Host "  reusing existing storage account: $STORAGE"
} else {
    az storage account create `
        --name $STORAGE --resource-group $RG --location $LOCATION `
        --sku Standard_LRS --kind StorageV2 --access-tier Hot | Out-Null
    Write-Host "  created storage account: $STORAGE"
}

$KEY = az storage account keys list -g $RG -n $STORAGE --query "[0].value" -o tsv
foreach ($c in @("corpus", "logits", "checkpoints", "tokenizer")) {
    az storage container create `
        --name $c --account-name $STORAGE --account-key $KEY | Out-Null
    Write-Host "  container ok: $c"
}

# ----------------------------- 3/6 AML workspace -----------------------------
Write-Host "[3/6] Azure ML workspace"
$wsList = az ml workspace list -g $RG --query "[].name" -o tsv
if ($wsList -and ($wsList -split "`n") -contains $WORKSPACE) {
    Write-Host "  workspace exists: $WORKSPACE"
} else {
    az ml workspace create --name $WORKSPACE --resource-group $RG `
        --location $LOCATION | Out-Null
    Write-Host "  created workspace: $WORKSPACE"
}

# ----------------------------- 4/6 compute cluster -----------------------------
Write-Host "[4/6] compute cluster (T4 spot)"
$cmpList = az ml compute list -g $RG -w $WORKSPACE --query "[].name" -o tsv
if ($cmpList -and ($cmpList -split "`n") -contains $COMPUTE) {
    Write-Host "  compute exists: $COMPUTE"
} else {
    az ml compute create `
        --name $COMPUTE --type AmlCompute `
        --size $VM_SIZE `
        --min-instances 0 --max-instances 1 `
        --tier LowPriority `
        --idle-time-before-scale-down $IDLE_SECS `
        --resource-group $RG --workspace-name $WORKSPACE | Out-Null
    Write-Host "  created compute: $COMPUTE"
}

# ----------------------------- 5/6 quota check -----------------------------
Write-Host "[5/6] quota check"
$usage = az vm list-usage --location $LOCATION -o json | ConvertFrom-Json
$t4 = $usage | Where-Object { $_.name.value -like "*NCASv3_T4*" } | Select-Object -First 1
if ($t4) {
    Write-Host ("  T4 quota:  used={0}  limit={1}" -f $t4.currentValue, $t4.limit)
    if ($t4.limit -lt 4) {
        Write-Warning "  T4 quota below 4 vCPU. Request a raise:"
        Write-Warning "  Portal -> Subscriptions -> Usage + quotas -> NCASv3_T4 Family vCPUs"
    }
} else {
    Write-Warning "  could not read T4 quota for $LOCATION"
}

$lowPri = $usage | Where-Object { $_.name.value -like "*lowPriorityCores*" } | Select-Object -First 1
if ($lowPri) {
    Write-Host ("  low-priority cores:  used={0}  limit={1}" -f $lowPri.currentValue, $lowPri.limit)
}

# ----------------------------- 6/6 budget alerts -----------------------------
Write-Host "[6/6] budget + alerts"
$start = (Get-Date -Format "yyyy-MM-01")
$end   = (Get-Date).AddMonths(6).ToString("yyyy-MM-01")

# az consumption budget create has shifting flags across CLI versions.
# Use ARM REST API for robustness.
$budgetUri = "/subscriptions/$SUB/providers/Microsoft.Consumption/budgets/$BUDGET_NAME" + `
             "?api-version=2023-05-01"
$body = @{
    properties = @{
        category    = "Cost"
        amount      = $BUDGET_USD
        timeGrain   = "Monthly"
        timePeriod  = @{
            startDate = $start + "T00:00:00Z"
            endDate   = $end   + "T00:00:00Z"
        }
        notifications = @{
            actual_50 = @{
                enabled = $true; operator = "GreaterThan"; threshold = 50
                contactEmails = @($EMAIL); thresholdType = "Actual"
            }
            actual_75 = @{
                enabled = $true; operator = "GreaterThan"; threshold = 75
                contactEmails = @($EMAIL); thresholdType = "Actual"
            }
            actual_90 = @{
                enabled = $true; operator = "GreaterThan"; threshold = 90
                contactEmails = @($EMAIL); thresholdType = "Actual"
            }
            forecast_100 = @{
                enabled = $true; operator = "GreaterThan"; threshold = 100
                contactEmails = @($EMAIL); thresholdType = "Forecasted"
            }
        }
    }
} | ConvertTo-Json -Depth 8

$bodyFile = New-TemporaryFile
$body | Out-File -FilePath $bodyFile -Encoding utf8
try {
    az rest --method put --uri $budgetUri --body "@$bodyFile" `
            --headers "Content-Type=application/json" | Out-Null
    Write-Host "  budget set: `$$BUDGET_USD / month with alerts at 50/75/90/100%"
} catch {
    Write-Warning "  budget API call failed: $_"
    Write-Warning "  set the budget manually: Portal -> Cost Management -> Budgets"
} finally {
    Remove-Item $bodyFile -Force -ErrorAction SilentlyContinue
}

# ----------------------------- summary -----------------------------
Write-Host ""
Write-Host "===================== done ====================="
Write-Host "  resource group : $RG"
Write-Host "  storage        : $STORAGE"
Write-Host "    containers   : corpus, logits, checkpoints, tokenizer"
Write-Host "  AML workspace  : $WORKSPACE"
Write-Host "  compute        : $COMPUTE ($VM_SIZE LowPriority 0..1)"
Write-Host "  budget         : `$$BUDGET_USD/mo  alerts -> $EMAIL"
Write-Host ""
Write-Host "next steps:"
Write-Host "  1. upload your corpus to blob:"
Write-Host "       az storage blob upload-batch -d corpus -s data\raw \\"
Write-Host "          --account-name $STORAGE --account-key <KEY>"
Write-Host "  2. submit a training job via Azure ML CLI v2 (job.yaml)"
Write-Host "  3. monitor cost: az consumption usage list --top 20"
Write-Host ""
Write-Host "save the storage key for upload steps:"
Write-Host "  `$env:AZURE_STORAGE_KEY = '$KEY'"
