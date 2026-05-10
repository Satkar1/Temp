# HP-FRM Integration — Technical Handbook

**Tickets:** EDPE-1954 / IC-3399  
**Purpose:** Single document for **manager briefing**, **live demo**, **code review**, and **onboarding**.  
**Companion:** Deeper flow narrative lives in [`HP_FRM_FLOW.md`](./HP_FRM_FLOW.md); this file adds **setup**, **endpoints**, **DB**, **demo script**, and **review checklist**.

---

## Table of contents

1. [Executive summary (for manager)](#1-executive-summary-for-manager)  
2. [Product behaviour in one page](#2-product-behaviour-in-one-page)  
3. [Architecture](#3-architecture)  
4. [Repositories and key files](#4-repositories-and-key-files)  
5. [API endpoints (actual paths)](#5-api-endpoints-actual-paths)  
6. [RMS configuration](#6-rms-configuration)  
7. [PayU configuration](#7-payu-configuration)  
8. [Database: what changes where](#8-database-what-changes-where)  
9. [Local / dotlocal setup (step-by-step)](#9-local--dotlocal-setup-step-by-step)  
10. [Mock Ultron (local HP API)](#10-mock-ultron-local-hp-api)  
11. [Logs and grep keywords](#11-logs-and-grep-keywords)  
12. [Demo script (what to show)](#12-demo-script-what-to-show)  
13. [Verification SQL](#13-verification-sql)  
14. [Code review checklist](#14-code-review-checklist)  
15. [Failure modes and timeouts](#15-failure-modes-and-timeouts)  
16. [Cold restart checklist](#16-cold-restart-checklist)

---

## 1. Executive summary (for manager)

- **Who:** HP merchants routed to **Ultron / HP FRM** for fraud assessment.  
- **What:** PayU already talks to **RMS** (Java). RMS talks to **Trident/Ultron**. For HP we add a **flag** and a **dedicated Ultron URL + instance headers**.  
- **Critical product rule:** On HP **DECLINE**, PayU **does not block** the customer before the bank; payment can **capture**. After **successful capture**, PayU **queues auto-refund** so money is returned. **ALERT** is **not** auto-refunded (HP ops).  
- **Visibility:** Decision surfaces on the transaction as **`udf4`** (ALLOW / ALERT / DECLINE) and **`udf5`** (rule name, e.g. MCM3), plus existing RMS audit (`txn_rms_info`).  
- **Scope:** Changes span **`riskmicroservice`** (routing + response fields) and **`payu`** (payload flag, UDF mapping, post-capture refund hook).

---

## 2. Product behaviour in one page

| Phase | When | What happens |
|--------|------|----------------|
| **A — Pre-PG** | Before redirect to bank | PayU calls RMS → RMS may call Ultron (HP URL). Response maps to **`udf4`/`udf5`**. Payment flow continues per RMS **`allowed`** semantics for HP (DECLINE still proceeds to PG — see PRD / `HP_FRM_FLOW.md`). |
| **B — Post-capture** | After txn is **CAPTURED** | If merchant has **`enableHPFRMTrident`** and **`udf4` = DECLINE**, PayU calls **`cancelBeforeSettling()`** → row in **`auto_refund_requests`**, status **`autoRefund`**. Cron / refund service drains the queue later. |

---

## 3. Architecture

```
User → PayU (PHP) ──POST JSON──► RMS (Spring Boot) ──POST──► Ultron / HP FRM
                      │                                  │
                      │                                  └── ruleSuggestion, observations
                      └── Saves riskAction, udf4, udf5 on transaction

Bank/PG → capture success → PayU → RiskManager::assessRiskPostSuccess()
         └── HP + DECLINE + CAPTURED → cancelBeforeSettling → auto_refund_requests
```

---

## 4. Repositories and key files

### PayU (`payu`)

| Area | Path | Responsibility |
|------|------|----------------|
| RMS payload | `lib/utility/PayuRisk.php` | Adds **`enableHPFRMTrident: true`** when merchant param set; builds RMS JSON; **`processRiskResponse()`** saves **`udf4`/`udf5`** from **`ruleSuggestion`** / **`declineRuleName`**; **`mapUltronDecisionToPayU()`**. |
| Post-capture refund | `lib/utility/RiskManager.php` | **`assessRiskPostSuccess()`**: HP + **`udf4`** DECLINE + CAPTURED → **`cancelBeforeSettling()`**. |
| Refund queue | `lib/model/Transaction.php` | **`cancelBeforeSettling()`** — inserts **`auto_refund_requests`**, sets **`STATUS_AUTO_REFUND`**. |
| Refund model | `lib/model/RefundRequests.php` | Table **`auto_refund_requests`**. |
| Risk entry | `lib/core/PaymentFlow.php` / `lib/model/Transaction.php` | Orchestration into **`PayuRisk`** / **`RiskManager`** (see `HP_FRM_FLOW.md`). |
| Checkout entry | `apps/payu/actions/_payment_options.php` | User reaches payment / risk gate. |

### RMS (`riskmicroservice`)

| Area | Path | Responsibility |
|------|------|----------------|
| HTTP API | `src/main/java/com/riskanalysisservice/controller/RiskAnalysisController.java` | **`POST /api/v1/riskAnalysis/validateTransaction`**. |
| Trident call | `src/main/java/com/riskanalysisservice/service/TridentService.java` | If **`enableHPFRMTrident`** → **`trident.hpFrmUrl`**, **`instanceId`** body, **`masterInstanceId`/`slaveInstanceId`** headers; parse **`ruleSuggestion`**, **`extractDeclineRuleName`**. |
| Final response | `src/main/java/com/riskanalysisservice/service/DecisionManager.java` | Copies **`ruleSuggestion`** / **`declineRuleName`** onto final **`Response`** (must not drop). |
| DTO | `src/main/java/com/riskanalysisservice/dto/Response.java` | Fields **`ruleSuggestion`**, **`declineRuleName`**. |
| Config bean | `src/main/java/com/riskanalysisservice/config/TridentConfig.java` (or equivalent package) | **`hpFrmUrl`**, **`hpFrmMasterInstanceId`**, **`hpFrmSlaveInstanceId`**. |
| Properties | `src/main/resources/application.properties` | **`trident.hpFrmUrl`**, **`server.port`**, access logs dir. |

### Mock (optional, local only)

| Path | Role |
|------|------|
| `riskmicroservice/scripts/mock_ultron.py` | Returns JSON **`DENY` + observations (MCM3)** or **`?mode=accept|alert`**. |

---

## 5. API endpoints (actual paths)

### RMS (default local port from `application.properties`)

Base: **`http://localhost:49153`**

| Purpose | Method | Full URL |
|---------|--------|----------|
| Health | GET | `http://localhost:49153/actuator/health` |
| PayU main risk call | POST | `http://localhost:49153/api/v1/riskAnalysis/validateTransaction` |
| Merchant params (admin tooling) | GET | `http://localhost:49153/api/v1/riskMicroservice/merchant-params/fetchMerchantParams?merchantId=<id>` |
| Param keys | GET | `http://localhost:49153/api/v1/riskMicroservice/merchant-params/fetchParamKeys` |

### Ultron / HP (production — illustrative)

RMS posts to whatever is configured in **`trident.hpFrmUrl`** (full URL including path).

### Mock Ultron (local)

| Purpose | URL |
|---------|-----|
| Health / help text | GET `http://127.0.0.1:48089/analyse/request` |
| Analyse (DENY default) | POST `http://127.0.0.1:48089/analyse/request` |
| Modes | POST `http://127.0.0.1:48089/analyse/request?mode=deny` / `?mode=alert` / `?mode=accept` |

### PayU → RMS when PayU runs **inside Docker**

The PHP container must reach RMS on the host:

| Config key | Example value |
|------------|----------------|
| **`riskServiceUrl`** | `http://host.docker.internal:49153/api/v1/riskAnalysis/validateTransaction` |

### PayU → RMS merchant-params base

| Config key | Example value |
|------------|----------------|
| **`risk_service_base_url`** | `http://host.docker.internal:49153/api/v1/riskMicroservice` |

*(PayU appends paths such as `/merchant-params/...` — the base must include **`/api/v1/riskMicroservice`**.)*

---

## 6. RMS configuration

**File:** `riskmicroservice/src/main/resources/application.properties`

| Property | Meaning |
|----------|---------|
| **`server.port`** | HTTP port (e.g. **49153**). |
| **`trident.hpFrmUrl`** | Full Ultron HP endpoint URL for **`enableHPFRMTrident`** traffic. |
| **`trident.hpFrm.masterInstanceId`** | Sent as body **`instanceId`** and header **`masterInstanceId`**. |
| **`trident.hpFrm.slaveInstanceId`** | Header **`slaveInstanceId`**. |
| **`server.undertow.accesslog.dir`** | RMS access logs directory (machine-specific path). |

**Build / run:**

```bash
cd /Users/satkar.garje/workspace/riskmicroservice
./gradlew bootJar -x test
java -jar build/libs/RiskAnalysisService-0.0.1-SNAPSHOT.jar
```

*(If JAR name differs: `ls build/libs`.)*

**Important:** Do not commit real Ultron URLs / credentials into shared branches without policy. Local mocks use **`localhost`** only.

---

## 7. PayU configuration

### Global config (MySQL)

Loaded via **`ConfigBase::get()`** — historically **`payu.config`** table (and mirrors such as **`payu.config_extra`** depending on env).

| Key | Purpose |
|-----|---------|
| **`riskServiceUrl`** | Full POST URL for **`validateTransaction`** (must match RMS host from PHP’s network — use **`host.docker.internal`** from Docker). |
| **`risk_service_base_url`** | Base URL for **`/api/v1/riskMicroservice/...`** calls. |
| **`tmxTimeout`** / **`riskifiedTimeout`** | When TMX/Riskified flags are sent, effective curl timeout may follow these (alongside **`tridentTimeout`** merchant param). Raise locally if RMS round-trip exceeds ~2s to avoid **`riskAction`** request-fail / empty HP fields. |

### Merchant parameters

Stored per merchant (e.g. **`merchant_param`** / admin — exact table name depends on PayU schema).

| Param | Purpose |
|-------|---------|
| **`mafEnabled`** | Risk / MAF path enabled (as per existing PayU rules). |
| **`enableHPFRMTrident`** | **`1`** / truthy → PayU sends **`enableHPFRMTrident: true`** to RMS; enables Phase B refund predicate. **`0`** / absent → HP path off. |
| **`tridentTimeout`** | Merchant-level RMS curl timeout (ms); interacts with global timeouts above. |

---

## 8. Database: what changes where

### No new HP-specific DDL required for core flow

HP reuses existing columns and tables.

### `payu.transaction` (read/write)

| Column / field | HP relevance |
|----------------|--------------|
| **`udf4`** | **`ALLOW`**, **`ALERT`**, or **`DECLINE`** (mapped from Ultron **`ruleSuggestion`**). |
| **`udf5`** | Rule name from RMS **`declineRuleName`** (e.g. **MCM3**); may be null on allow paths. |
| **`riskAction`** | Existing PayU risk bitmask / enums after RMS (e.g. deny vs accept paths). |
| **`status`** | **`autoRefund`** after **`cancelBeforeSettling()`** when DECLINE + captured. |
| **`addedon`** / **`updatedon`** | **`addedon`** = txn creation (often IST-style string); **`updatedon`** updated on saves — compares cleanly to **`auto_refund_requests.created_at`** when diagnosing timing. |

### `payu.auto_refund_requests`

| Column | Typical value for HP DECLINE |
|--------|-------------------------------|
| **`payu_id`** | PayU transaction id |
| **`source`** | **`cancelBeforeSettling`** |
| **`request`** | JSON including **`reference_id`** / **`token`** like **`cbset_<payu_id>`** |
| **`status`** | Pending until cron/refund service processes |

### `payu.txn_rms_info` (aggregated DB)

PayU **`PayuRisk::saveDetailsInTable()`** inserts RMS outcome audit (**`payu_id`**, **`risk_action`**, **`response_code`**, **`risk_service_used`**, **`extra_params`**).

### `payu.config` / `payu.config_extra`

Rows for **`riskServiceUrl`**, **`risk_service_base_url`**, timeout keys — **not** HP-named keys; they affect whether RMS is reachable within curl timeout.

---

## 9. Local / dotlocal setup (step-by-step)

1. **Start infrastructure** (Docker): e.g. **`dotlocal-mysql`**, **`dotlocal-web`**, etc.  
2. **Apply DB config** (MySQL), using **`docker exec -i dotlocal-mysql mysql ...`** if applicable:
   - Set **`riskServiceUrl`** and **`risk_service_base_url`** for Docker→host RMS.
   - Optionally raise **`tmxTimeout`** / **`riskifiedTimeout`** for slow local RMS + mock.
3. **Merchant:** enable **`mafEnabled`**, **`enableHPFRMTrident`**, sensible **`tridentTimeout`**.  
4. **Start mock Ultron** (optional):  
   `python3 /Users/satkar.garje/workspace/riskmicroservice/scripts/mock_ultron.py --port 48089`  
5. **Start RMS** with **`trident.hpFrmUrl=http://localhost:48089/analyse/request`** (mock on host).  
6. **Smoke:**
   - `curl -sS http://localhost:49153/actuator/health`
   - `curl -sS http://127.0.0.1:48089/analyse/request`

---

## 10. Mock Ultron (local HP API)

**Path:** `/Users/satkar.garje/workspace/riskmicroservice/scripts/mock_ultron.py`

| Mode | `ruleSuggestion` | Notes |
|------|------------------|--------|
| **`deny`** (default) | **`DENY`** | Observations include **MCM3** → RMS derives **`declineRuleName`**. |
| **`alert`** | **`ACCEPT_AND_ALERT`** | Alert path testing. |
| **`accept`** | **`ACCEPT_1FA`** | Allow-style testing. |

**Run:**

```bash
cd /Users/satkar.garje/workspace/riskmicroservice
python3 scripts/mock_ultron.py --host 0.0.0.0 --port 48089
```

RMS must point **`trident.hpFrmUrl`** at the **same path** the mock serves: **`/analyse/request`**.

---

## 11. Logs and grep keywords

### RMS

| Location | Notes |
|----------|--------|
| **`logs/access.*.log`** under **`server.undertow.accesslog.dir`** | JSON lines; **`requestURI`**, **`status`**, timings. |
| Terminal running **`java -jar`** | Stack traces, **`Request body prepared for Trident API`**, etc. |

**Search for:** `validateTransaction`, `tridentResponseTime`, `merchant-params`, `403`, `500`, `transactionId`, `HP-FRM`, `ruleSuggestion`.

### PayU

| Location | Notes |
|----------|--------|
| **`docker logs -f dotlocal-web`** | When PHP runs in Docker. |
| App log files | Depends on **`LOGGER_PATH`** / deployment (filter **`Risk Service`**, **`PayuRisk`**, **`RMS Exception`**). |

**Search for:** `Risk Service`, `Risk Service Curl`, `denied the transaction`, `PayuId`, `cancelBeforeSettling`, `auto_refund_requests`, `HP-FRM auto-refund marking failed`.

---

## 12. Demo script (what to show)

1. **Config slide:** Merchant **`enableHPFRMTrident`**, PayU **`riskServiceUrl`** → RMS; RMS **`trident.hpFrmUrl`** → Ultron or mock.  
2. **Trigger txn** with HP flag on (card flow / mode as configured).  
3. **Show RMS access log** line for **`POST /api/v1/riskAnalysis/validateTransaction`** — **200**, latency.  
4. **Show PayU DB row:** **`udf4`** = DECLINE, **`udf5`** = MCM3 (mock deny), **`riskAction`** consistent with deny-but-proceed semantics.  
5. **Complete capture** (or show already-captured txn): **`status`** → **`autoRefund`**.  
6. **Show `auto_refund_requests`**: **`source`** = **`cancelBeforeSettling`**, **`request`** contains **`cbset_<id>`**.  
7. **Contrast:** Turn **`enableHPFRMTrident`** off or use **`?mode=accept`** on mock — **`udf4`** ALLOW, no auto-refund queue for decline path.

---

## 13. Verification SQL

```sql
-- Latest HP-ish txns
SELECT id, merchantid, status, riskAction, udf4, udf5, addedon, updatedon
FROM payu.transaction
ORDER BY id DESC
LIMIT 5;

-- Refund queue
SELECT id, payu_id, source, status, try_count, created_at, updated_at, LEFT(`request`, 200)
FROM payu.auto_refund_requests
ORDER BY id DESC
LIMIT 5;

-- Config sanity (keys may vary by env)
SELECT `key`, LEFT(`value`, 120)
FROM payu.config
WHERE `key` IN ('riskServiceUrl', 'risk_service_base_url', 'tmxTimeout', 'riskifiedTimeout');
```

**Timezone note:** **`transaction.addedon`** is often PHP-written wall time (e.g. IST); **`auto_refund_requests.created_at`** is MySQL **`TIMESTAMP`** — compare **`transaction.updatedon`** to **`created_at`** for same-event correlation.

---

## 14. Code review checklist

### PayU

- [ ] **`PayuRisk.php`**: Flag only sent when **`enableHPFRMTrident`** truthy; absent/`0` unchanged for other merchants.  
- [ ] **`processRiskResponse()`**: Saves **`udf4`/`udf5`** when **`ruleSuggestion`** or **`declineRuleName`** present; mapping matches PRD.  
- [ ] **`mapUltronDecisionToPayU()`**: ACCEPT*/ALERT paths vs DECLINE; unknown values surfaced uppercased.  
- [ ] **`RiskManager::assessRiskPostSuccess()`**: Guards — HP enabled, **`udf4`** DECLINE, CAPTURED, not already auto-refund; try/catch logs failures without breaking payment.  
- [ ] **`cancelBeforeSettling()`**: Idempotency / duplicate handling (**`23000`** log) understood.  
- [ ] No regression: non-HP merchants’ payloads not bloated; timeouts documented.

### RMS

- [ ] **`TridentService`**: URL selection — HP vs domestic/intl/BFL; headers only on HP branch.  
- [ ] **`details: true`** (if present) for observations — needed for **`declineRuleName`**.  
- [ ] **`extractDeclineRuleName`**: Lowest **`ratingAdded`**, tie-break; fallback nesting per BOU sample.  
- [ ] **`Response`**: JSON serialization exposes **`ruleSuggestion`** / **`declineRuleName`**.  
- [ ] **`DecisionManager`**: HP fields copied to **final** response (easy to miss).  
- [ ] Config externalized — no hardcoded prod URLs in Java.

### Ops / security

- [ ] Secrets not committed (tokens, Accertify, keys in **`application.properties`** — review locally).  
- [ ] Ultron TLS / allowlists for prod.

---

## 15. Failure modes and timeouts

| Symptom | Likely cause |
|---------|----------------|
| **`riskAction`** request-fail / empty **`udf4`** | PayU curl **timeout** shorter than RMS round-trip; tune **`tmxTimeout`**, **`riskifiedTimeout`**, **`tridentTimeout`**. |
| RMS **403** on merchant-params | Auth / routing separate from **`validateTransaction`** — check Kong/admin client. |
| No row in **`auto_refund_requests`** | Predicates not met (not CAPTURED, not DECLINE, HP flag off), or DB connectivity on insert; wrong table (** not **`transaction_refund`** immediately). |
| Mock not hit | Wrong **`trident.hpFrmUrl`** path/port; RMS on host vs Docker networking confusion. |

---

## 16. Cold restart checklist

| Step | Command / action |
|------|------------------|
| Docker | Start dotlocal stack; confirm **`dotlocal-mysql`**, **`dotlocal-web`**. |
| MySQL | Re-verify **`riskServiceUrl`**, **`risk_service_base_url`**, timeouts. |
| Mock | `python3 scripts/mock_ultron.py --port 48089` |
| RMS | `./gradlew bootJar -x test && java -jar build/libs/RiskAnalysisService-0.0.1-SNAPSHOT.jar` |
| Smoke | `curl http://localhost:49153/actuator/health` and mock GET |

**Background processes (typical):**

```bash
# Terminal 1 — mock
cd /Users/satkar.garje/workspace/riskmicroservice && python3 scripts/mock_ultron.py --port 48089

# Terminal 2 — RMS
cd /Users/satkar.garje/workspace/riskmicroservice && java -jar build/libs/RiskAnalysisService-0.0.1-SNAPSHOT.jar
```

---

## Document maintenance

- When Wibmo confirms **production Ultron URL** and **instance IDs**, update **`application.properties`** and the **Executive summary** slide — remove mock-only wording.  
- Keep **`HP_FRM_FLOW.md`** as the long-form sequence / sequence-diagram source; update **this file** when onboarding steps or ports change.
