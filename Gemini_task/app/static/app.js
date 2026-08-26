// ==========================================================================
// Gemini Task - Interactive Dashboard JS Controller (Bootstrap 5 Standardized)
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    // API 基礎路徑
    const API_BASE = "/scheduler";

    // DOM 元素引用
    const llmPrompt = document.getElementById("llm-prompt");
    const btnParseLlm = document.getElementById("btn-parse-llm");
    const btnParseText = document.getElementById("btn-parse-text");
    const btnParseSpinner = document.getElementById("btn-parse-spinner");
    const llmErrorMsg = document.getElementById("llm-error-msg");

    const quotaValue = document.getElementById("quota-value");
    const quotaProgressFill = document.getElementById("quota-progress-fill");

    const statRunning = document.getElementById("stat-running");
    const statPending = document.getElementById("stat-pending");
    const statCompleted = document.getElementById("stat-completed");
    const statFailed = document.getElementById("stat-failed");

    const manualForm = document.getElementById("manual-task-form");
    const taskTypeSelect = document.getElementById("task-type");
    const triggerTypeSelect = document.getElementById("trigger-type");
    const intervalDaysInput = document.getElementById("interval-days");

    const tabActiveBtn = document.getElementById("tab-active-btn");
    const tabCompletedBtn = document.getElementById("tab-completed-btn");
    const tabFailedBtn = document.getElementById("tab-failed-btn");
    const jobsTableBody = document.getElementById("jobs-table-body");

    // Modal 相關元素
    const confirmModalEl = document.getElementById("confirm-modal");
    let bsConfirmModal = null;
    if (confirmModalEl && typeof bootstrap !== "undefined") {
        bsConfirmModal = new bootstrap.Modal(confirmModalEl);
    }

    const modalName = document.getElementById("modal-name");
    const modalTaskType = document.getElementById("modal-task-type");
    const modalTriggerType = document.getElementById("modal-trigger-type");
    const modalIntervalDays = document.getElementById("modal-interval-days");
    const modalTriggerTime = document.getElementById("modal-trigger-time");
    const modalRemarks = document.getElementById("modal-remarks");
    const modalConfirmBtn = document.getElementById("modal-confirm-btn");

    // 當前全域狀態
    let currentTab = "active"; // active, completed, failed
    let allJobs = [];

    // ----------------- 初始化與事件綁定 -----------------

    // 觸發類型變更時，啟用/停用更新週期天數
    if (triggerTypeSelect && intervalDaysInput) {
        triggerTypeSelect.addEventListener("change", () => {
            if (triggerTypeSelect.value === "auto") {
                intervalDaysInput.removeAttribute("disabled");
                intervalDaysInput.setAttribute("required", "true");
                intervalDaysInput.value = intervalDaysInput.value || 1;
            } else {
                intervalDaysInput.setAttribute("disabled", "true");
                intervalDaysInput.removeAttribute("required");
                intervalDaysInput.value = "";
            }
        });
    }

    if (modalTriggerType && modalIntervalDays) {
        modalTriggerType.addEventListener("change", () => {
            if (modalTriggerType.value === "auto") {
                modalIntervalDays.removeAttribute("disabled");
                modalIntervalDays.value = modalIntervalDays.value || 1;
            } else {
                modalIntervalDays.setAttribute("disabled", "true");
                modalIntervalDays.value = "";
            }
        });
    }

    // Tab 切換事件
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            currentTab = e.target.dataset.tab;
            renderTable();
        });
    });

    // 任務手動建立表單提交
    if (manualForm) {
        manualForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            const payload = {
                name: document.getElementById("task-name").value,
                task_type: taskTypeSelect.value,
                trigger_type: triggerTypeSelect.value,
                trigger_time: document.getElementById("trigger-time").value || null,
                interval_days: triggerTypeSelect.value === "auto" ? parseInt(intervalDaysInput.value) : null,
                remarks: document.getElementById("task-remarks").value || null
            };

            try {
                const resp = await fetch(`${API_BASE}/api/jobs`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || "建立任務失敗");
                }

                // 重置表單並刷新列表
                manualForm.reset();
                if (intervalDaysInput) intervalDaysInput.setAttribute("disabled", "true");
                fetchJobs();
                alert("任務建立成功！");
            } catch (err) {
                alert(err.message);
            }
        });
    }

    // AI 解析按鈕事件
    if (btnParseLlm && llmPrompt) {
        btnParseLlm.addEventListener("click", async () => {
            const promptText = llmPrompt.value.trim();
            if (!promptText) {
                showLlmError("請先輸入您的排程描述需求。");
                return;
            }

            clearLlmError();
            setLlmLoading(true);

            try {
                const resp = await fetch(`${API_BASE}/api/llm/parse`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: promptText })
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || "AI 解析失敗。");
                }

                const parsedResult = await resp.json();
                showConfirmModal(parsedResult);
                fetchQuota(); // 成功解析後，刷新配額顯示
            } catch (err) {
                showLlmError(err.message);
            } finally {
                setLlmLoading(false);
            }
        });
    }

    // Modal 確認送出事件
    if (modalConfirmBtn) {
        modalConfirmBtn.addEventListener("click", async () => {
            const payload = {
                name: modalName.value,
                task_type: modalTaskType.value,
                trigger_type: modalTriggerType.value,
                trigger_time: modalTriggerTime.value || null,
                interval_days: modalTriggerType.value === "auto" ? parseInt(modalIntervalDays.value) : null,
                remarks: modalRemarks.value || null
            };

            try {
                const resp = await fetch(`${API_BASE}/api/jobs`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || "建立任務失敗");
                }

                closeModal();
                if (llmPrompt) llmPrompt.value = ""; // 清空 AI 對話框
                fetchJobs();
                alert("已成功由 AI 解析並加入排程任務！");
            } catch (err) {
                alert(err.message);
            }
        });
    }

    // ----------------- 功能函式 -----------------

    function setLlmLoading(isLoading) {
        if (!btnParseLlm) return;
        if (isLoading) {
            btnParseLlm.setAttribute("disabled", "true");
            if (btnParseSpinner) btnParseSpinner.classList.remove("d-none");
            if (btnParseText) btnParseText.classList.add("d-none");
        } else {
            btnParseLlm.removeAttribute("disabled");
            if (btnParseSpinner) btnParseSpinner.classList.add("d-none");
            if (btnParseText) btnParseText.classList.remove("d-none");
        }
    }

    function showLlmError(msg) {
        if (!llmErrorMsg) return;
        llmErrorMsg.textContent = `錯誤：${msg}`;
        llmErrorMsg.classList.remove("d-none");
    }

    function clearLlmError() {
        if (!llmErrorMsg) return;
        llmErrorMsg.textContent = "";
        llmErrorMsg.classList.add("d-none");
    }

    function showConfirmModal(data) {
        if (modalName) modalName.value = data.name || "";
        if (modalTaskType) modalTaskType.value = data.task_type || "tw_stock_cost";
        if (modalTriggerType) modalTriggerType.value = data.interval_days ? "auto" : "llm";
        
        if (modalIntervalDays) {
            if (data.interval_days) {
                modalIntervalDays.removeAttribute("disabled");
                modalIntervalDays.value = data.interval_days;
            } else {
                modalIntervalDays.setAttribute("disabled", "true");
                modalIntervalDays.value = "";
            }
        }
        
        if (modalTriggerTime) modalTriggerTime.value = data.trigger_time || "";
        if (modalRemarks) modalRemarks.value = data.remarks || "";

        if (bsConfirmModal) {
            bsConfirmModal.show();
        } else if (confirmModalEl) {
            confirmModalEl.classList.add("show");
            confirmModalEl.style.display = "block";
        }
    }

    function closeModal() {
        if (bsConfirmModal) {
            bsConfirmModal.hide();
        } else if (confirmModalEl) {
            confirmModalEl.classList.remove("show");
            confirmModalEl.style.display = "none";
        }
    }

    // 格式化時間
    function formatTime(isoStr) {
        if (!isoStr) return "-";
        try {
            const d = new Date(isoStr);
            if (isNaN(d.getTime())) return "-";
            return d.getFullYear() + '-' +
                String(d.getMonth() + 1).padStart(2, '0') + '-' +
                String(d.getDate()).padStart(2, '0') + ' ' +
                String(d.getHours()).padStart(2, '0') + ':' +
                String(d.getMinutes()).padStart(2, '0');
        } catch {
            return "-";
        }
    }

    // 格式化任務類型顯示名稱
    function getTaskTypeName(type) {
        const typesMap = {
            "tw_stock_cost": "台股股價更新",
            "us_stock_cost": "美股股價更新",
            "tw_stock_price_only": "更新全台灣股價",
            "us_stock_price_only": "更新全美國股價",
            "twse_investor": "台股上市三大法人",
            "tpex_investor": "台股上櫃三大法人",
            "us_investor": "美股法人持股",
            "tw_listed_list_update": "台灣上市公司清單更新",
            "tw_otc_list_update": "台灣上櫃公司清單更新",
            "us_stock_list_update": "美國上市公司清單更新",
            "update_macro_data_us": "更新美國總體經濟數據",
            "update_macro_data_tw": "更新台灣總體經濟數據"
        };
        return typesMap[type] || type;
    }

    // 格式化耗時
    function formatDuration(sec) {
        if (sec === null || sec === undefined) return "-";
        if (sec < 60) return `${sec.toFixed(1)} 秒`;
        const min = Math.floor(sec / 60);
        const remSec = sec % 60;
        return `${min} 分 ${remSec.toFixed(0)} 秒`;
    }

    // 立即觸發任務
    window.triggerImmediately = async function(id) {
        if (!confirm("確定要立即重新執行此任務嗎？")) return;
        try {
            const resp = await fetch(`${API_BASE}/api/jobs/execute/${id}`, { method: "POST" });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "觸發失敗");
            }
            fetchJobs();
            alert("任務已重設為 pending，即將在背景執行！");
        } catch (err) {
            alert(err.message);
        }
    };

    // 物理刪除或取消排程
    window.deleteJob = async function(id, status) {
        const actionText = status === "pending" ? "刪除此排程任務" : "將此任務標記為取消";
        if (!confirm(`確定要${actionText}嗎？`)) return;
        try {
            const resp = await fetch(`${API_BASE}/api/jobs/${id}`, { method: "DELETE" });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "操作失敗");
            }
            fetchJobs();
        } catch (err) {
            alert(err.message);
        }
    };

    // 取得配額
    async function fetchQuota() {
        try {
            const resp = await fetch(`${API_BASE}/api/llm/usage`);
            if (resp.ok) {
                const data = await resp.json();
                if (quotaValue) quotaValue.textContent = `${data.count} / ${data.limit}`;
                const pct = Math.min((data.count / data.limit) * 100, 100);
                if (quotaProgressFill) {
                    quotaProgressFill.style.width = `${pct}%`;
                    if (pct >= 90) {
                        quotaProgressFill.className = "progress-bar bg-danger";
                    } else if (pct >= 75) {
                        quotaProgressFill.className = "progress-bar bg-warning text-dark";
                    } else {
                        quotaProgressFill.className = "progress-bar bg-primary";
                    }
                }
            }
        } catch (err) {
            console.error("無法載入 LLM 配額資訊: ", err);
        }
    }

    // 取得所有任務並計算統計
    async function fetchJobs() {
        try {
            const resp = await fetch(`${API_BASE}/api/jobs`);
            if (resp.ok) {
                allJobs = await resp.json();
                updateStats();
                renderTable();
            }
        } catch (err) {
            console.error("載入任務列表失敗: ", err);
        }
    }

    // 更新指標卡片
    function updateStats() {
        let running = 0;
        let pending = 0;
        let completed = 0;
        let failed = 0;

        allJobs.forEach(job => {
            if (job.status === "running") running++;
            else if (job.status === "pending") pending++;
            else if (job.status === "completed") completed++;
            else if (job.status === "failed" || job.status === "cancelled") failed++;
        });

        if (statRunning) statRunning.textContent = running;
        if (statPending) statPending.textContent = pending;
        if (statCompleted) statCompleted.textContent = completed;
        if (statFailed) statFailed.textContent = failed;
    }

    // 渲染任務表格
    function renderTable() {
        if (!jobsTableBody) return;

        let filteredJobs = [];
        if (currentTab === "active") {
            filteredJobs = allJobs.filter(job => job.status === "pending" || job.status === "running");
        } else if (currentTab === "completed") {
            filteredJobs = allJobs.filter(job => job.status === "completed");
        } else if (currentTab === "failed") {
            filteredJobs = allJobs.filter(job => job.status === "failed" || job.status === "cancelled");
        }

        if (filteredJobs.length === 0) {
            jobsTableBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted py-4">目前沒有符合此狀態的任務。</td></tr>`;
            return;
        }

        jobsTableBody.innerHTML = filteredJobs.map(job => {
            let badgeClass = "badge-pending";
            let statusText = "排程中";
            if (job.status === "running") {
                badgeClass = "badge-running";
                statusText = "執行中";
            } else if (job.status === "completed") {
                badgeClass = "badge-completed";
                statusText = "已完成";
            } else if (job.status === "failed") {
                badgeClass = "badge-failed";
                statusText = "失敗";
            } else if (job.status === "cancelled") {
                badgeClass = "badge-cancelled";
                statusText = "已取消";
            }

            let triggerSource = "手動";
            if (job.trigger_type === "auto") {
                triggerSource = `定期 (每 ${job.interval_days} 天)`;
            } else if (job.trigger_type === "llm") {
                triggerSource = "AI 解析";
            }

            let remarksHtml = "-";
            if (job.remarks) {
                remarksHtml = `<span class="remarks-text" title="${job.remarks.replace(/"/g, '&quot;')}">${job.remarks}</span>`;
            }

            let actionsHtml = "";
            if (job.status === "pending") {
                actionsHtml = `
                    <button class="btn-icon text-success" onclick="triggerImmediately(${job.id})" title="立即執行">
                        <i class="bi bi-play-fill"></i>
                    </button>
                    <button class="btn-icon btn-icon-danger text-danger" onclick="deleteJob(${job.id}, 'pending')" title="刪除任務">
                        <i class="bi bi-trash-fill"></i>
                    </button>
                `;
            } else if (job.status === "running") {
                actionsHtml = `<span class="text-info spinner-border spinner-border-sm" role="status"></span>`;
            } else {
                actionsHtml = `
                    <button class="btn-icon text-primary" onclick="triggerImmediately(${job.id})" title="重新觸發">
                        <i class="bi bi-arrow-counterclockwise"></i>
                    </button>
                `;
                if (job.status !== "cancelled") {
                    actionsHtml += `
                        <button class="btn-icon btn-icon-danger text-secondary" onclick="deleteJob(${job.id}, '${job.status}')" title="取消">
                            <i class="bi bi-slash-circle"></i>
                        </button>
                    `;
                }
            }

            return `
                <tr>
                    <td><span class="text-muted small">${job.id}</span></td>
                    <td>
                        <div class="fw-semibold" title="${job.name.replace(/"/g, '&quot;')}">${job.name}</div>
                        ${remarksHtml !== "-" ? `<div class="mt-1">${remarksHtml}</div>` : ""}
                    </td>
                    <td><span class="badge ${badgeClass}">${getTaskTypeName(job.task_type)}</span></td>
                    <td><span class="small text-muted">${triggerSource}</span></td>
                    <td><span class="small">${formatTime(job.trigger_time)}</span></td>
                    <td><span class="small">${formatTime(job.completion_time)}</span></td>
                    <td><span class="small">${formatDuration(job.duration)}</span></td>
                    <td>
                        <div class="d-flex gap-1">
                            ${actionsHtml}
                        </div>
                    </td>
                </tr>
            `;
        }).join("");
    }

    // ----------------- 定時刷新任務 -----------------
    fetchQuota();
    fetchJobs();
    
    // 每 5 秒自動重新整理任務列表與配額
    setInterval(() => {
        fetchJobs();
    }, 5000);

    // 每 30 秒重新整理配額
    setInterval(() => {
        fetchQuota();
    }, 30000);
});
