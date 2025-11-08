// ============================
// Fake News Detection JS Logic
// Author: Pratima Sahu
// ============================

async function analyzeNews() {
  const text = document.getElementById("newsText").value.trim();
  const box = document.getElementById("resultBox");
  const spinner = document.getElementById("spinner");

  if (!text) {
    alert("Please enter some text first!");
    return;
  }

  box.style.display = "block";
  spinner.style.display = "inline-block";
  box.innerHTML = `<b>Analyzing...</b>`;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();

    spinner.style.display = "none";

    // Color result based on prediction
    const color = data.label === "FAKE" ? "#b91c1c" : "#047857";
    box.style.color = color;
    box.innerHTML = `<b>Prediction:</b> ${data.label}<br><b>Confidence:</b> ${data.confidence}%`;

  } catch (error) {
    console.error("Prediction Error:", error);
    box.style.color = "#dc2626";
    box.innerHTML = "Error: Unable to connect to server!";
    spinner.style.display = "none";
  }
}

// Clear function
function clearText() {
  document.getElementById("newsText").value = "";
  document.getElementById("resultBox").style.display = "none";
}
