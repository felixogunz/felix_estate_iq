const $ = (selector) => document.querySelector(selector);

const form = $("#prediction-form");
const stateSelect = $("#state");
const townSelect = $("#town");
const titleSelect = $("#title");
const predictButton = $("#predict-button");
const buttonLabel = $("#button-label");
const errorMessage = $("#form-error");

let townsByState = {};
let metricsChart = null;

const formatNaira = (value) => {
    if (!Number.isFinite(Number(value))) return "—";
    return new Intl.NumberFormat("en-NG", {
        style: "currency",
        currency: "NGN",
        maximumFractionDigits: 0,
    }).format(Number(value));
};

const formatCompactNaira = (value) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";

  if (number >= 1_000_000_000) {
    return `₦${(number / 1_000_000_000).toFixed(2)}B`;
  }
  if (number >= 1_000_000) {
    return `₦${(number / 1_000_000).toFixed(1)}M`;
  }
  if (number >= 1_000) {
    return `₦${(number / 1_000).toFixed(1)}K`;
  }
  return formatNaira(number);
};

function populateSelect(select, values, placeholder) {
  select.replaceChildren();

  const firstOption = document.createElement("option");
  firstOption.value = "";
  firstOption.textContent = placeholder;
  select.appendChild(firstOption);

  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

function updateTowns() {
  const selectedState = stateSelect.value;
  const towns = townsByState[selectedState] || [];

  populateSelect(
    townSelect,
    towns,
    towns.length ? "Choose a town / area" : "No towns available"
  );

  townSelect.disabled = towns.length === 0;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.error || data.message || `Request failed (${response.status})`);
  }

  return data;
}

async function loadOptions() {
  const data = await fetchJson("/api/options");

  townsByState = data.towns_by_state || {};

  populateSelect(stateSelect, data.states || [], "Choose a state");
  populateSelect(titleSelect, data.property_types || [], "Choose a property type");

  $("#listing-value").textContent = Number(data.listing_count || 0).toLocaleString("en-NG");

  if (data.states?.length) {
    const preferredState = data.states.find(
      (state) => state.toLowerCase() === "lagos"
    ) || data.states[0];

    stateSelect.value = preferredState;
    updateTowns();
  }
}

function drawMetricsChart(metrics) {
  const canvas = $("#metrics-chart");
  if (!canvas || typeof Chart === "undefined") {
    $("#chart-note").textContent =
      "Chart library unavailable. See the metric cards above for model results.";
    return;
  }

  if (metricsChart) metricsChart.destroy();

  const maeMillions = Number(metrics.mae_naira) / 1_000_000;
  const rmseMillions = Number(metrics.rmse_naira) / 1_000_000;

  metricsChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["MAE", "RMSE"],
      datasets: [{
        label: "Error in millions of naira",
        data: [maeMillions, rmseMillions],
        backgroundColor: [
          "rgba(80, 226, 189, .78)",
          "rgba(173, 154, 255, .78)",
        ],
        borderColor: ["#50e2bd", "#ad9aff"],
        borderWidth: 1,
        borderRadius: 8,
        maxBarThickness: 58,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 900,
        easing: "easeOutQuart",
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => ` ${formatCompactNaira(context.raw * 1_000_000)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#a7bdc9", font: { family: "DM Sans" } },
          border: { color: "rgba(147,190,207,.15)" },
        },
        y: {
          beginAtZero: true,
          grid: { color: "rgba(147,190,207,.09)" },
          ticks: {
            color: "#8099a9",
            callback: (value) => `₦${value}M`,
            font: { family: "DM Sans", size: 10 },
          },
          border: { display: false },
        },
      },
    },
  });
}

async function loadMetrics() {
  const data = await fetchJson("/api/metrics");

  if (!data.ready) {
    $("#mae-value").textContent = "—";
    $("#r2-value").textContent = "—";
    $("#chart-note").textContent =
      "Train the model before displaying performance metrics.";
    return;
  }

  $("#mae-value").textContent = formatCompactNaira(data.mae_naira);
  $("#r2-value").textContent = Number(data.r2).toFixed(3);
  $("#model-status").textContent = "Trained";
  $("#model-status-foot").textContent =
    `${Number(data.test_rows || 0).toLocaleString("en-NG")} test listings`;

  $("#chart-note").textContent =
    `Test-set performance · ${Number(data.test_rows || 0).toLocaleString("en-NG")} held-out listings.`;

  drawMetricsChart(data);
}

async function checkHealth() {
  const status = $("#model-status");
  const foot = $("#model-status-foot");

  try {
    const data = await fetchJson("/api/health");

    if (data.model_ready) {
      status.textContent = "Online";
      foot.textContent = "Model file detected";
    } else {
      status.textContent = "Not trained";
      foot.textContent = "Run the training step";
    }
  } catch {
    status.textContent = "Offline";
    foot.textContent = "Could not reach the API";
  }
}

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  buttonLabel.textContent = isLoading ? "Analysing property…" : "Generate price estimate";
  predictButton.setAttribute("aria-busy", String(isLoading));
}

function updateResult(data, input) {
  const price = Number(data.estimated_price_ngn);

  $("#predicted-price").textContent = formatNaira(price);
  $("#result-caption").textContent =
    "Model estimate from the available Nigerian property listings.";

  $("#summary-location").textContent = `${input.town}, ${input.state}`;
  $("#summary-type").textContent = input.title;
  $("#summary-bedrooms").textContent = input.bedrooms;

  $("#result-disclaimer").textContent = data.disclaimer ||
    "This estimate is based on historical advertised listings, not a formal valuation.";

  const stage = $("#result-stage");
  stage.classList.remove("price-pop");
  void stage.offsetWidth;
  stage.classList.add("price-pop");

  $("#predicted-price").setAttribute("aria-label", `Estimated price ${formatNaira(price)}`);
}

stateSelect.addEventListener("change", updateTowns);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.textContent = "";

  if (!form.reportValidity()) return;

  const input = {
    state: stateSelect.value,
    town: townSelect.value,
    title: titleSelect.value,
    bedrooms: Number($("#bedrooms").value),
    bathrooms: Number($("#bathrooms").value),
    toilets: Number($("#toilets").value),
    parking_space: Number($("#parking_space").value),
  };

  setLoading(true);

  try {
    const result = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });

    updateResult(result, input);
  } catch (error) {
    errorMessage.textContent = error.message ||
      "Unable to calculate an estimate. Please try again.";
  } finally {
    setLoading(false);
  }
});

// Reveal sections as they enter the viewport.
const revealElements = document.querySelectorAll(".reveal");

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  revealElements.forEach((element) => observer.observe(element));
} else {
  revealElements.forEach((element) => element.classList.add("is-visible"));
}

// Highlight the current section in the navigation.
const navLinks = [...document.querySelectorAll(".nav-link")];

if ("IntersectionObserver" in window) {
  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;

      navLinks.forEach((link) => {
        link.classList.toggle(
          "active",
          link.getAttribute("href") === `#${entry.target.id}`
        );
      });
    });
  }, { rootMargin: "-20% 0px -65% 0px" });

  ["home", "predict", "insights", "about"].forEach((id) => {
    const section = document.getElementById(id);
    if (section) sectionObserver.observe(section);
  });
}

async function initialiseDashboard() {
  try {
    await loadOptions();
  } catch (error) {
    errorMessage.textContent =
      `Could not load property options: ${error.message}`;
    populateSelect(stateSelect, [], "Dataset unavailable");
    populateSelect(titleSelect, [], "Dataset unavailable");
  }

  try {
    await loadMetrics();
  } catch (error) {
    $("#chart-note").textContent = `Could not load metrics: ${error.message}`;
  }

  await checkHealth();
}

initialiseDashboard();
