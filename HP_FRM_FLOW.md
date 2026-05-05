# HP-FRM (EDPE-1954 / IC-3399) — End-to-End Technical Documentation

> **Scope:** Integrate HP (Hewlett-Packard) merchants with a dedicated FRM (fraud rules) flow on Trident/Ultron via PayU's existing RMS gateway. PayU does not block payments on HP DECLINE during pre-PG; instead it captures, then auto-refunds DECLINE-flagged transactions post-capture. ALERT decisions are not auto-refunded (handled by HP ops dashboard).

---

## Table of Contents
1. [Two-Phase Mental Model](#1-two-phase-mental-model)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase A — Pre-PG Risk Validation](#3-phase-a--pre-pg-risk-validation)
4. [RMS Routing & HP Branch](#4-rms-routing--hp-branch)
5. [Response Handling on PayU](#5-response-handling-on-payu)
6. [Payment Proceeds to PG / Bank](#6-payment-proceeds-to-pg--bank)
7. [Phase B — Post-Capture Auto-Refund](#7-phase-b--post-capture-auto-refund)
8. [Refund Cron Pipeline](#8-refund-cron-pipeline)
9. [Merchant Visibility (Webhooks / Verify)](#9-merchant-visibility-webhooks--verify)
10. [Configuration Prerequisites](#10-configuration-prerequisites)
11. [Failure Modes & Fallbacks](#11-failure-modes--fallbacks)
12. [Hardcoded-for-Testing Inventory](#12-hardcoded-for-testing-inventory)
13. [Local Testing Guide](#13-local-testing-guide)
14. [What's Remaining](#14-whats-remaining)
15. [Sequence Diagram](#15-sequence-diagram)
16. [Files Touched & Diff Summary](#16-files-touched--diff-summary)

---

## 1. Two-Phase Mental Model

The whole feature splits into two phases with clear responsibilities:

| Phase | When | Purpose | Result |
|---|---|---|---|
| **A — Pre-PG risk** | Before user is sent to bank/PG | Get HP risk decision; do **not** block the payment if HP says DECLINE | UDF4/UDF5 saved on txn |
| **B — Post-capture refund** | After bank captures money | If HP said DECLINE, refund the captured txn | Auto-refund queued + cron drains |

```
[user] → [PayU Phase A: risk] → [PG/Bank capture] → [PayU Phase B: post-success risk] → [refund cron]
                  │                                                    │
                  ▼                                                    ▼
              [RMS]                                               (no RMS call here)
                  │
                  ▼
              [Trident / Ultron HP]
```

---

## 2. Architecture Overview

```
┌─────────┐  HTTPS  ┌──────┐  HTTPS  ┌──────────────┐
│ Browser │────────▶│ PayU │────────▶│ RMS          │
│  /User  │         │ PHP  │         │ (Java SBoot) │
└─────────┘         └───┬──┘         └──────┬───────┘
                        │                   │
                  redirect│                 │ HP branch
                        ▼                   ▼
                  ┌────────┐         ┌──────────────┐
                  │ PG/Bank│         │ Trident/Ultron│
                  │        │         │   (HP FRM)    │
                  └────┬───┘         └──────────────┘
                       │
                  webhook
                       ▼
                  ┌────────┐  cron   ┌─────────────────┐
                  │ PayU PHP├────────▶│ RefundService   │
                  │ (Phase B)        │ (initiate refund)│
                  └────────┘         └─────────────────┘
```

---

## 3. Phase A — Pre-PG Risk Validation

### 3.1 Entry Point

The merchant POSTs payment params to PayU. The action handler is `apps/payu/actions/_payment_options.php`. The flow is then orchestrated by `PaymentFlow::execute()` / `::initiatePayment()` / `::makePayment()` (`lib/core/PaymentFlow.php` — 8290 lines).

The flow eventually reaches the **risk gate** which is what triggers Phase A through `Transaction::assessRiskOnProcessingPayment()` in `Transaction.php`.

### 3.2 Building the RMS Request — *PayU edit #1*

**File:** `payu/lib/utility/PayuRisk.php` (around line 95–120)

```php
$rmsPayload = [
    "transactionId" => $transaction->payu_id,
    "merchantId"    => $transaction->merchantid,
    "purchaseAmount"=> $transaction->amount,
    "mode"          => $transaction->mode,
    "firstName"     => $transaction->firstname,
    ...
];
if (!empty($_COOKIE['USERTXNINFO'])) {
    $rmsPayload["cookie_id"] = $_COOKIE['USERTXNINFO'];
}

// HP-FRM: Enable Ultron/HP routing in RMS for HP MIDs only.
$enableHpFrm = $merchant->getMerchantParam('enableHPFRMTrident');
if (!empty($enableHpFrm) && (string)$enableHpFrm !== '0') {
    $rmsPayload["enableHPFRMTrident"] = true;
}
```

**Behavior:**
- Reads `enableHPFRMTrident` merchant param.
- If truthy → adds `"enableHPFRMTrident": true` to the RMS payload.
- If absent / `0` → flag is **not** sent (zero impact for non-HP merchants).

### 3.3 Sending to RMS

`PayuRisk::doService()` POSTs the JSON to:

```
POST {riskServiceUrl}/api/v1/riskAnalysis/validateTransaction
Content-Type: application/json
```

PayU treats RMS as best-effort — if curl fails or times out, PayU falls back to `allowed=true` (no regression vs today).

---

## 4. RMS Routing & HP Branch

### 4.1 Controller Entry

**File:** `riskmicroservice/.../controller/RiskAnalysisController.java`

```java
@RestController
@RequestMapping("/api/v1/riskAnalysis")
public class RiskAnalysisController {
    @PostMapping("/validateTransaction")
    public ResponseEntity<Response> performRiskAnalysis(@RequestBody String request, ...) {
        ...
        ResponseEntity<Response> response = riskAnalysisService.analyseRisk(requestJsonObject, httpServletRequest);
```

### 4.2 Service Orchestration

`RiskAnalysisService.analyseRisk()` performs:
1. `validateRequest()` — mandatory-field checks.
2. Decides which engines to call (Trident, Riskified, Rambo).
3. Calls `TridentService.analyseTransaction()`.
4. Hands all engine results to `DecisionManager` for the final decision.

### 4.3 Trident Routing — *RMS edit #2 (TridentService.java)*

```java
private Response sendRequestToTrident(boolean isInternational, int timeoutThreshold,
        JSONObject requestPayload, HttpServletRequest httpServletRequest) throws JsonProcessingException {

    boolean enableHpFrm = requestPayload.optBoolean("enableHPFRMTrident", false);   // ① read flag
    URI tridentUrl = enableHpFrm
        ? tridentConfig.getHpFrmUrl()                         // ② HP-FRM branch
        : (isInternational
            ? tridentConfig.getInternationUrl()
            : (isBflTransaction ? tridentConfig.getBflTridentUrl()
                                : tridentConfig.getDomesticUrl()));

    httpServletRequest.setAttribute(Constants.CHANNEL_ID, requestPayload.get(Constants.CHANNEL_ID));
    long tridentAPIEvaluationStartTime = System.currentTimeMillis();

    // HP-FRM (Ultron) request decoration: per Postman, master orchestrates and slave is HP.
    HttpHeaders requestHeaders = new HttpHeaders();
    if (enableHpFrm) {
        requestPayload.put("instanceId", tridentConfig.getHpFrmMasterInstanceId());      // ③ body
        requestHeaders.set("masterInstanceId", tridentConfig.getHpFrmMasterInstanceId()); // ④ header
        requestHeaders.set("slaveInstanceId", tridentConfig.getHpFrmSlaveInstanceId());   // ⑤ header
    }

    try {
        log.info("Request body prepared for Trident API: {}", requestPayload);
        tridentResponse = restTemplate.exchange(tridentUrl, HttpMethod.POST,
                new HttpEntity<>(requestPayload.toMap(), requestHeaders), String.class);
```

**What happens for HP requests:**
- ① Read `enableHPFRMTrident` flag.
- ② URL switches to `hpFrmUrl` (Ultron host).
- ③ Body inject: `instanceId = master`.
- ④/⑤ Headers: `masterInstanceId`, `slaveInstanceId`.

For non-HP requests, behavior is **identical** to before (empty `HttpHeaders` is a no-op).

### 4.4 Config — *RMS edit #3 (TridentConfig.java)*

```java
@Configuration
@Data
public class TridentConfig {
    @Value("${trident.internationalUrl}")  private URI internationUrl;
    @Value("${trident.domesticUrl}")       private URI domesticUrl;
    @Value("${DS_SALT}")                   private String dsSalt;
    @Value("${trident.bfl.domesticUrl}")   private URI bflTridentUrl;
    @Value("${trident.bfl.merchantid}")    private String bflMerchantId;

    // HP-FRM (Ultron) config. URL + instance IDs are placeholders until Wibmo/infra confirms.
    @Value("${trident.hpFrmUrl:}")               private URI    hpFrmUrl;
    @Value("${trident.hpFrm.masterInstanceId:}") private String hpFrmMasterInstanceId;
    @Value("${trident.hpFrm.slaveInstanceId:}")  private String hpFrmSlaveInstanceId;
}
```

### 4.5 Properties (placeholder values)

```properties
# HP-FRM (Ultron) — PLACEHOLDERS for testing. Replace once Wibmo/infra confirms UAT/Prod values.
trident.hpFrmUrl=https://uat-ultron.example.payu.in/ultron/analyse/risk
trident.hpFrm.masterInstanceId=PAYU_MASTER_PLACEHOLDER
trident.hpFrm.slaveInstanceId=HP_SLAVE_PLACEHOLDER
```

When Wibmo confirms, only these three values change.

### 4.6 Outbound HP HTTP Call

```
POST https://{ultron-host}/ultron/analyse/risk
Content-Type: application/json
masterInstanceId: PAYU_MASTER_PLACEHOLDER
slaveInstanceId: HP_SLAVE_PLACEHOLDER

{
  "transactionId": "...",
  "merchantId": "...",
  "instanceId": "PAYU_MASTER_PLACEHOLDER",
  ... (all other fields RMS already builds for Trident)
  "enableHPFRMTrident": true
}
```

### 4.7 Reading HP Fields from Ultron Response

```java
String hpDecision = null;
String hpReason   = null;
if (enableHpFrm && tridentResponse != null && tridentResponse.getBody() != null) {
  try {
    JSONObject body = new JSONObject(tridentResponse.getBody());
    // TODO: fallback "ALLOW" / null are placeholders so PayU can be tested
    //       before Wibmo adds the real fields. Drop fallbacks once real schema lands.
    hpDecision = body.optString("hpRiskDecision",  "ALLOW");
    hpReason   = body.optString("hpDeclineReason", null);
  } catch (Exception ignore) { }
}

return Response.builder()
    .code(HttpStatus.OK.value())
    ...
    .hpRiskDecision(hpDecision)
    .hpDeclineReason(hpReason)
    .build();
```

### 4.8 Response DTO — *RMS edit #4 (Response.java)*

```java
@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class Response {
    private int    code;
    private String message;
    private Object result;
    private boolean isAllowed;
    private int    tridentResponseTime;
    private boolean allowTxnFor1Fa;
    private String riskEngine;
    private String hpRiskDecision;     // TODO(Wibmo): rename if Wibmo confirms a different field
    private String hpDeclineReason;    // TODO(Wibmo): rename if Wibmo confirms a different field
}
```

### 4.9 DecisionManager Propagation — *RMS edit #5 (DecisionManager.java)*

`fetchFinalResponse()` builds a fresh `Response` for the 1FA / 2FA / deny branch. We must explicitly carry HP fields through:

```java
String riskEngineHonoured = fetchHonouredRiskEngine(...);
Response finalResponse    = fetchFinalResponse(result, riskEngineResponse, transactionId, riskEngineHonoured);
finalResponse.setHpRiskDecision(tridentResponse.getHpRiskDecision());
finalResponse.setHpDeclineReason(tridentResponse.getHpDeclineReason());
return finalResponse;
```

Without these two lines, HP fields would silently get dropped at this boundary — subtle but critical.

---

## 5. Response Handling on PayU

### 5.1 Parsing & UDF Save — *PayU edit #6*

**File:** `payu/lib/utility/PayuRisk.php` (around line 336–358)

```php
try {
    $result = [];
    if ($response[CurlBase::RESPONSE_KEY_CURL_STATUS] == DataArray::CURL_SUCCESS) {
        $result = json_decode($response[CurlBase::RESPONSE_KEY_RESULT], true);
    } else {
        $result['code']         = 500;
        $result['message']      = 'Exception occurred while risk validation';
        $result['allowed']      = true;
        $result['allowTxnFor1Fa']= false;
    }

    // HP-FRM: RMS returns hpRiskDecision/hpDeclineReason for HP merchants.
    // Per product direction, HP fields overwrite udf4/udf5 unconditionally for HP txns
    // (collision with merchant-supplied UDFs / BNPL udf5 is acceptable in HP scope).
    if (is_array($result)) {
        $hpDecision = $result['hpRiskDecision'] ?? null;
        $hpReason   = $result['hpDeclineReason'] ?? null;
        if (!empty($hpDecision) || !empty($hpReason)) {
            $transaction->save([
                'udf4' => $hpDecision,
                'udf5' => $hpReason
            ], false);
        }
    }
```

### 5.2 Why UDF4 / UDF5?

1. **Already echoed in webhooks** — `Merchant::callbackws` and `verify_payment` API include UDFs by default.
2. **HP product agreed** — per PRD direction.
3. **Phase B reads them** — post-success refund logic reads `udf4`, no extra plumbing.

### 5.3 Continuing the Normal Risk Flow

After the HP UDF save, `processRiskResponse()` proceeds as usual:
- Computes `transactionIsAllowed`, `allowFor1FA`.
- Sets `riskEngineHonoured` extra-param.
- Returns service response to `RiskManager`.

`RiskManager` decides deny / 2FA-challenge / pass — but **HP DECLINE does not deny here**. The HP rule is: let it go to PG; refund post-capture.

---

## 6. Payment Proceeds to PG / Bank

Standard PayU PG flow:
- `CheckOutUtility` / `PaymentFlow` chooses the gateway.
- User redirected to bank (3DS / OTP / etc.).
- Bank returns success → PG webhook to PayU → PayU updates txn status:
  - `STATUS_AUTH` (auth only)
  - `STATUS_CAPTURED` (auth + captured)

The capture handler eventually calls back into the post-success assessment.

---

## 7. Phase B — Post-Capture Auto-Refund

### 7.1 Trigger

The capture/success path in `PaymentFlow.php` calls `RiskManager::assessRiskPostSuccess()`.

### 7.2 Auto-Refund Trigger — *PayU edit #7*

**File:** `payu/lib/utility/RiskManager.php` (around line 258–273)

```php
// HP-FRM: For HP merchants, if the risk engine suggests DECLINE we still allow the payment
// to proceed, but after capture-success we must mark the txn for auto-refund.
// ALERT is intentionally not auto-refunded (handled by HP ops via dashboard).
try {
    $enableHpFrm       = $merchant->getMerchantParam('enableHPFRMTrident');
    $isHpFrmEnabled    = !empty($enableHpFrm) && (string)$enableHpFrm !== '0';
    $isDecline         = !empty($this->transaction->udf4) &&
                         strcasecmp($this->transaction->udf4, 'DECLINE') === 0;
    $isCaptured        = $this->transaction->status === Transaction::STATUS_CAPTURED;
    $alreadyAutoRefund = $this->transaction->status === Transaction::STATUS_AUTO_REFUND;

    if ($isHpFrmEnabled && $isDecline && $isCaptured && !$alreadyAutoRefund) {
        $this->transaction->cancelBeforeSettling(null, true);
    }
} catch (Exception $e) {
    Logger::log("HP-FRM auto-refund marking failed for PayuId: "
                . $this->transaction->getId() . " Error: " . $e->getMessage());
}
```

### 7.3 Predicate Matrix

| Flag | UDF4 | Status | Action |
|---|---|---|---|
| Off | * | * | no-op (non-HP txn) |
| On | ALLOW | captured | no-op |
| On | ALERT | captured | no-op (per PRD) |
| On | DECLINE | captured | **`cancelBeforeSettling()`** ← refund queued |
| On | DECLINE | already autoRefund | no-op (idempotency) |
| On | DECLINE | not captured | no-op (only refund money that's actually captured) |

### 7.4 What `cancelBeforeSettling()` Does

Existing PayU primitive in `Transaction.php`:
1. Inserts a row into `auto_refund_requests` (unique key per `payu_id` → idempotency).
2. Flips txn status to `STATUS_AUTO_REFUND`.
3. Marks the txn as block-from-settlement so PayU doesn't pay the merchant for it.

**No actual refund API call yet** — just queued.

---

## 8. Refund Cron Pipeline

### 8.1 Cron Entry

The batch job `batch/processAutoRefundRequests.php` runs on schedule and:
1. Reads pending rows from `auto_refund_requests`.
2. For each, calls `RefundRequests::autoRefundAPI()` (in `lib/model/RefundRequests.php`).
3. That hits the **refund microservice**: `POST /refund/v1/initiate`.

### 8.2 Refund Microservice → Bank

Refund microservice handles the actual reversal with the issuing bank/network. PayU webhook to the merchant fires when the refund succeeds.

### 8.3 Settlement Block

`cancelBeforeSettling()` flagged the txn for settlement-block, so MDR/settlement engines won't pay the merchant for this txn — even if refund takes time, the merchant doesn't get the money first.

---

## 9. Merchant Visibility (Webhooks / Verify)

### 9.1 Webhook (`Merchant::callbackws`)

PayU fires an S2S webhook to the merchant on state transitions. The default payload **includes UDFs**, so HP merchants automatically see:
- `udf4=DECLINE` / `udf5=<rule_name>` on initial completion (Phase A result).
- Status transitions (`captured` → `autoRefund` → `refunded`) on subsequent webhooks.

### 9.2 Verify Payment API

Same — `verify_payment` returns UDFs, so HP merchants can poll if needed.

---

## 10. Configuration Prerequisites

| Where | Config | Purpose | Status |
|---|---|---|---|
| PayU merchant config | `enableHPFRMTrident=1` for HP MIDs | Master toggle per merchant | **TODO: provision** |
| RMS `application.properties` | `trident.hpFrmUrl` | Real Ultron URL | Placeholder; awaiting Wibmo |
| RMS `application.properties` | `trident.hpFrm.masterInstanceId` | Master instance ID | Placeholder; awaiting Wibmo |
| RMS `application.properties` | `trident.hpFrm.slaveInstanceId` | Slave (HP) instance ID | Placeholder; awaiting Wibmo |

If `enableHPFRMTrident` is missing on a merchant → HP code is fully inert for that merchant.

---

## 11. Failure Modes & Fallbacks

| Failure | Behavior |
|---|---|
| RMS unreachable | PayU `processRiskResponse` falls back to `allowed=true`, no UDF write, no auto-refund |
| Ultron unreachable | RMS catches `ResourceAccessException`, returns 504 to PayU, HP fields null, no UDF write |
| HP fields missing in RMS response | PayU's `if (!empty($hpDecision) ‖ !empty($hpReason))` skips the save — no accidental UDF wipe |
| `cancelBeforeSettling` throws | Caught and logged in `RiskManager::assessRiskPostSuccess`; payment is NOT failed because of refund-marking issues |
| Cron picks up same row twice | `auto_refund_requests` unique key prevents duplicate insert; `STATUS_AUTO_REFUND` guard prevents double-trigger |

---

## 12. Hardcoded-for-Testing Inventory

These are **intentional placeholders** so PayU↔RMS↔Trident path can be E2E tested before Wibmo/infra hand over real values. None are "missing implementation" — they are stubs in code/config that **must** be swapped when external team confirms.

### RMS

| # | Where | Value | Replace with |
|---|---|---|---|
| 1 | `application.properties` → `trident.hpFrmUrl` | `https://uat-ultron.example.payu.in/ultron/analyse/risk` | Real Ultron UAT/Prod URL (Wibmo) |
| 2 | `application.properties` → `trident.hpFrm.masterInstanceId` | `PAYU_MASTER_PLACEHOLDER` | Real master instance ID (Wibmo) |
| 3 | `application.properties` → `trident.hpFrm.slaveInstanceId` | `HP_SLAVE_PLACEHOLDER` | Real slave instance ID (Wibmo) |
| 4 | `TridentService.java` ~line 487 — fallback `body.optString("hpRiskDecision", "ALLOW")` | default `"ALLOW"` | Drop the fallback once Ultron returns the real field consistently |
| 5 | `TridentService.java` ~line 488 — fallback `body.optString("hpDeclineReason", null)` | default `null` | Map to real Ultron field path (likely lowest-risk-rating ruleName per PRD) |
| 6 | `Response.java` field names `hpRiskDecision` / `hpDeclineReason` | placeholder names | Rename if Wibmo's real schema uses different keys |

### PayU

| # | Where | Status |
|---|---|---|
| 7 | Merchant config → `enableHPFRMTrident` not yet provisioned in DB/admin | TODO once HP MIDs identified |

### NOT hardcoded (real code, no placeholder)

- HP routing logic (`enableHpFrm` branch in `sendRequestToTrident`).
- HP fields propagation in `DecisionManager.fetchFinalResponse`.
- PayU `udf4`/`udf5` save in `PayuRisk::processRiskResponse`.
- PayU auto-refund trigger in `RiskManager::assessRiskPostSuccess` (uses existing `cancelBeforeSettling` pipeline).

---

## 13. Local Testing Guide

### 13.1 Level 0 — Compile sanity (~30s)

```bash
cd /Users/satkar.garje/workspace/riskmicroservice
./gradlew compileJava
```

### 13.2 Level 1 — Boot RMS, observe HP routing

Requires PayU VPN/network for Redis/Kafka/MySQL.

```bash
./gradlew bootJar -x test
java -jar build/libs/RiskAnalysisService-0.0.1-SNAPSHOT.jar
# RMS comes up on http://localhost:49153
```

Health: `curl http://localhost:49153/actuator/health`.

Send an HP-flagged validateTransaction request (full payload format, with `"enableHPFRMTrident": true`). Logs will show:
- `Request body prepared for Trident API: {...}` with `instanceId` field
- `I/O error on POST request for "https://uat-ultron.example.payu.in/ultron/analyse/risk"` (DNS fails because URL is fake placeholder)
- Final response includes `hpRiskDecision=null, hpDeclineReason=null`

This proves HP branch wiring is live.

### 13.3 Level 2 — Mock Ultron locally, get success response

Spin up a Python mock:

```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        print("MOCK ULTRON HEADERS:", dict(self.headers))
        print("MOCK ULTRON BODY:", body.decode())
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
          "clientId": "TEST_CLIENT",
          "hpRiskDecision": "DECLINE",
          "hpDeclineReason": "RULE_XYZ_LOW_RATING"
        }).encode())
HTTPServer(("127.0.0.1", 8089), H).serve_forever()
```

Update `application.properties`:
```properties
trident.hpFrmUrl=http://127.0.0.1:8089/ultron/analyse/risk
```

Restart RMS and re-fire the HP-flagged request. Final response will include `hpRiskDecision=DECLINE, hpDeclineReason=RULE_XYZ_LOW_RATING`.

### 13.4 Level 3 — End-to-end with PayU

Run PayU locally, point its RMS URL config at `http://localhost:49153/...`. Set `enableHPFRMTrident=1` for a test merchant. Run a txn → observe `udf4=DECLINE`, `udf5=RULE_XYZ_LOW_RATING` saved on the txn → force capture → `cancelBeforeSettling` triggers `auto_refund_requests` row → cron initiates refund.

---

## 14. What's Remaining

### 14.1 Real Code/Config TODOs (you control)

- **PayU**: Provision `enableHPFRMTrident=1` for HP MIDs in merchant config / admin / DB.
- **RMS**: Tests — unit tests on `TridentService` HP branch + DTO test on `Response` + `DecisionManager` test for HP propagation.
- **PayU**: Tests — PHPUnit for `PayuRisk::processRiskResponse` HP block + `RiskManager::assessRiskPostSuccess` HP branch.
- **RMS**: HP-tagged logging (e.g. `[HP-FRM]` prefix) for ops grep-ability.
- **Decision needed**: `/ultron/update/status` leg (post-finalization update). Per Postman it exists but not yet wired. Confirm with product whether v1 needs it.

### 14.2 Awaiting External Team

- Real Ultron URL (Wibmo)
- Real master/slave instance IDs (Wibmo)
- Confirmed Ultron response field names and decline-reason format

### 14.3 QA / Verification (no code, just sign-off)

- Cron `processAutoRefundRequests` enabled in HP env.
- HP MID payment modes not in `NON_AUTO_REFUND_MODES` / `blockAutoRefundForEnach`.
- Webhook / verify API surface `udf4` / `udf5` for HP MIDs.
- E2E UAT all 6 cells of the predicate matrix (Section 7.3).
- Idempotency: simulate `assessRiskPostSuccess` running twice → only one `auto_refund_requests` row.

### 14.4 Optional Follow-ups

- Per-env kill switch on RMS side (so SRE can disable HP branch instantly without redeploy).
- Merchant integration doc update (UDF semantics for HP MIDs).
- Decision-string normalization (PayU compares `'DECLINE'` case-insensitively; if Wibmo emits `Reject`/`DENY`/etc., add a small mapping table on RMS).

---

## 15. Sequence Diagram

```
┌──────┐ ┌──────┐ ┌──────┐ ┌─────────┐ ┌─────────┐ ┌──────┐ ┌─────────┐ ┌────────┐
│Merch.│ │ PayU │ │ RMS  │ │Trident  │ │Ultron HP│ │  PG  │ │auto_ref.│ │Refund  │
│ /User│ │ PHP  │ │      │ │(domestic│ │         │ │/Bank │ │_requests│ │Service │
└──┬───┘ └──┬───┘ └──┬───┘ └────┬────┘ └────┬────┘ └──┬───┘ └────┬────┘ └────┬───┘
   │ POST  │        │           │            │         │          │            │
   │ txn   │        │           │            │         │          │            │
   ├──────►│        │           │            │         │          │            │
   │       │ POST validateTxn  │            │         │          │            │
   │       │ + enableHPFRM=true│            │         │          │            │
   │       ├───────►│           │            │         │          │            │
   │       │        │ HP branch │            │         │          │            │
   │       │        │ /ultron/analyse/risk   │         │          │            │
   │       │        │ instanceId+headers     │         │          │            │
   │       │        ├──────────────────────►│         │          │            │
   │       │        │                       │ orchestrate         │            │
   │       │        │ {hpRiskDecision:DECLINE, hpDeclineReason:R} │            │
   │       │        │◄──────────────────────┤         │          │            │
   │       │ Response{hpRiskDecision,hpDeclineReason} │          │            │
   │       │◄───────┤           │            │         │          │            │
   │       │ udf4=DECLINE,     │            │         │          │            │
   │       │ udf5=R, save txn  │            │         │          │            │
   │       │                   │            │         │          │            │
   │ redir.│                   │            │         │          │            │
   │◄──────┤ → PG              │            │         │          │            │
   │ pays  │                   │            │         │          │            │
   ├─────────────────────────────────────────────────►│          │            │
   │                            │            │         │          │            │
   │ ◄──────────────────────── PG webhook (success) ──┤          │            │
   │       │ assessRiskPostSuccess          │         │          │            │
   │       │ udf4==DECLINE && captured      │         │          │            │
   │       │ ─→ cancelBeforeSettling        │         │          │            │
   │       ├───────────────────────────────────────────────►│            │
   │       │                                                │            │
   │ ◄──── webhook(captured)                                │            │
   │       │                                                │            │
   │       │  ── cron tick ──                              │            │
   │       ├────────────────────────────────────────────────┤            │
   │       │ pickup pending rows                            │            │
   │       │ initiate refund                                ├───────────►│
   │       │                                                │            │
   │       │◄────────────── refund OK ─────────────────────────────────┤
   │ ◄──── webhook(refunded)                                │            │
```

---

## 16. Files Touched & Diff Summary

### RMS (branch `IC-3399`)

| File | Change |
|---|---|
| `src/main/java/com/riskanalysisservice/dto/Response.java` | Added `hpRiskDecision`, `hpDeclineReason` fields |
| `src/main/java/com/riskanalysisservice/config/TridentConfig.java` | Added `hpFrmUrl`, `hpFrmMasterInstanceId`, `hpFrmSlaveInstanceId` |
| `src/main/java/com/riskanalysisservice/service/TridentService.java` | HP routing branch: URL switch, `instanceId` body, `master`/`slave` instance ID headers; added `HttpHeaders` import |
| `src/main/java/com/riskanalysisservice/service/DecisionManager.java` | Copies `hpRiskDecision`/`hpDeclineReason` from `tridentResponse` onto `finalResponse` |
| `src/main/resources/application.properties` | Added 3 placeholder properties: `trident.hpFrmUrl`, `trident.hpFrm.masterInstanceId`, `trident.hpFrm.slaveInstanceId` |

### PayU

| File | Change |
|---|---|
| `lib/utility/PayuRisk.php` | (a) `createRiskServiceRequest()` — adds `enableHPFRMTrident: true` to RMS payload when merchant flag on. (b) `processRiskResponse()` — saves `hpRiskDecision` → `udf4`, `hpDeclineReason` → `udf5` |
| `lib/utility/RiskManager.php` | `assessRiskPostSuccess()` — auto-refund trigger when HP enabled + `udf4=DECLINE` + status=captured + not already autoRefund |

---

## 17. Three Critical Gotchas We Solved

1. **`HttpHeaders` import** — was missing; without it, the headers code wouldn't compile. Resolved by adding `import org.springframework.http.HttpHeaders;`.
2. **DecisionManager dropping HP fields** — `fetchFinalResponse()` builds a fresh `Response` per result branch; without explicit copy, our HP fields would be reset to null at the boundary. Fixed by copying after `fetchFinalResponse()`.
3. **UDF4/UDF5 unconditional overwrite** — avoided clever conditional logic per HP product direction. Simpler = safer.

---

## 18. References

- **Jira:** EDPE-1954 (Product), IC-3399 (Engineering branch)
- **Postman collection:** Trident Ultron `POST /ultron/analyse/risk`, `POST /ultron/update/status`
- **PRD:** "Decline reason = ruleName with lowest risk rating, alphabetical"
- **Master/slave orchestration rule:** "If master decides DECLINE, that wins" (Ultron returns orchestrated decision)

---

*Last updated: May 2026. Authors: Satkar Garje.*
