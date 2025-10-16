// // =========================
// // Symptom Categories
// // =========================
// const symptoms = {
//     general: ["Fever", "Fatigue", "Headache", "Body Pain", "Dizziness"],
//     respiratory: ["Cough", "Cold", "Sore Throat", "Shortness of Breath", "Chest Pain"],
//     digestive: ["Nausea", "Vomiting", "Diarrhea", "Stomach Pain", "Loss of Appetite"],
//     skin: ["Rash", "Itching", "Redness", "Swelling"]
// };

// // =========================
// // Populate Symptom Categories
// // =========================
// document.addEventListener("DOMContentLoaded", () => {
//     const categorySelect = document.getElementById("category");
//     if (categorySelect) {
//         Object.keys(symptoms).forEach(cat => {
//             const option = document.createElement("option");
//             option.value = cat;
//             option.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
//             categorySelect.appendChild(option);
//         });
//     }
// });

// // =========================
// // Show Symptoms Dynamically
// // =========================
// function showSymptoms() {
//     const category = document.getElementById("category").value;
//     const container = document.getElementById("symptoms-list");
//     container.innerHTML = "";

//     if (category && symptoms[category]) {
//         symptoms[category].forEach(symptom => {
//             const wrapper = document.createElement("div");
//             wrapper.classList.add("form-check");

//             const checkbox = document.createElement("input");
//             checkbox.type = "checkbox";
//             checkbox.classList.add("form-check-input");
//             checkbox.name = "symptoms";
//             checkbox.value = symptom;

//             const label = document.createElement("label");
//             label.classList.add("form-check-label");
//             label.textContent = symptom;

//             wrapper.appendChild(checkbox);
//             wrapper.appendChild(label);
//             container.appendChild(wrapper);
//         });
//     }
// }

// // =========================
// // Manual Symptom Submission
// // =========================
// function getSymptomRecommendation() {
//     const checkedSymptoms = Array.from(document.querySelectorAll("input[name='symptoms']:checked"))
//         .map(symptom => symptom.value);

//     if (checkedSymptoms.length === 0) {
//         alert("⚠️ Please select at least one symptom.");
//         return;
//     }

//     const formData = { symptoms: checkedSymptoms };

//     fetch("/get_symptom_recommendation", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(formData),
//     })
//         .then(res => res.json())
//         .then(data => {
//             const resultDiv = document.getElementById("symptom-result");
//             resultDiv.classList.remove("d-none");

//             if (data.error) {
//                 resultDiv.classList.add("alert-danger");
//                 resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
//             } else {
//                 const medsList = Array.isArray(data.recommendation)
//                     ? data.recommendation.join(", ")
//                     : data.recommendation;

//                 resultDiv.classList.remove("alert-danger");
//                 resultDiv.classList.add("alert-info");
//                 resultDiv.innerHTML = `
//                     <h5><strong>Manual Symptom Results</strong></h5>
//                     <p><strong>Possible Cause:</strong> ${data.possible_cause}</p>
//                     <p><strong>Recommended Medicine:</strong> ${medsList}</p>
//                 `;
//             }
//         })
//         .catch(err => {
//             alert("Error: " + err);
//         });
// }

// function getPossibleCause() {
//     getSymptomRecommendation();
// }

// // =========================
// // Questionnaire Recommendation
// // =========================
// function getQuestionnaireRecommendation() {
//     const age = document.getElementById("age").value.trim();
//     const gender = document.getElementById("gender").value.trim();
//     const duration = document.getElementById("duration").value.trim();
//     const previousTreatment = document.getElementById("previous_treatment").value.trim();
//     const takingMedicine = document.getElementById("Taking_Medicine").value.trim();
//     const symptomsStr = document.getElementById("symptoms").value.trim();

//     if (!age || !gender || !symptomsStr) {
//         alert("⚠️ Please provide age, gender, and symptoms.");
//         return;
//     }

//     const formData = {
//         age,
//         gender,
//         symptoms: symptomsStr,
//         duration,
//         previous_treatment: previousTreatment,
//         taking_medicine: takingMedicine,
//     };

//     fetch("/get_questionnaire_recommendation", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(formData),
//     })
//         .then(res => res.json())
//         .then(data => {
//             const resultDiv = document.getElementById("questionnaire-result");
//             resultDiv.classList.remove("d-none");

//             if (data.error) {
//                 resultDiv.classList.add("alert-danger");
//                 resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
//             } else {
//                 const medsList = Array.isArray(data.recommendation)
//                     ? data.recommendation.join(", ")
//                     : data.recommendation;

//                 resultDiv.classList.remove("alert-danger");
//                 resultDiv.classList.add("alert-info");
//                 resultDiv.innerHTML = `
//                     <h5><strong>Questionnaire Results</strong></h5>
//                     <p><strong>Predicted Disease:</strong> ${data.disease}</p>
//                     <p><strong>Possible Cause:</strong> ${data.possible_cause}</p>
//                     <p><strong>Recommended Medicine:</strong> ${medsList}</p>
//                 `;
//             }
//         })
//         .catch(err => {
//             alert("Error: " + err);
//         });
// }

// // =========================
// // Image Prediction Upload
// // =========================
// function uploadImageAndPredict(modelType) {
//     const form = document.getElementById("image-form");
//     const imageInput = document.getElementById("image");
//     const resultDiv = document.getElementById("image-analysis-result");

//     if (imageInput.files.length === 0) {
//         alert("⚠️ Please select an image first.");
//         return;
//     }

//     const formData = new FormData(form);
//     formData.append("model_type", modelType);

//     resultDiv.classList.remove("d-none");
//     resultDiv.innerHTML = `<p class="text-info">⏳ Processing image... Please wait.</p>`;

//     fetch("/upload_and_predict", {
//         method: "POST",
//         body: formData,
//     })
//         .then(res => res.json())
//         .then(data => {
//             if (data.error) {
//                 resultDiv.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
//             } else {
//                 const medsList = Array.isArray(data.recommendation)
//                     ? data.recommendation.join(", ")
//                     : data.recommendation;

//                 resultDiv.innerHTML = `
//                     <h5><strong>Image Analysis Results</strong></h5>
//                     <p><strong>Predicted Condition:</strong> ${data.possible_cause}</p>
//                     <p><strong>Recommendation:</strong> ${medsList}</p>
//                     <div class="mt-3 text-center">
//                         <img src="${data.path || document.getElementById('preview').src}" 
//                              alt="Uploaded Image" 
//                              class="img-fluid rounded shadow-sm" 
//                              style="max-width: 300px; border-radius: 10px;">
//                     </div>
//                 `;
//             }
//         })
//         .catch(err => {
//             resultDiv.innerHTML = `<p class="text-danger">Failed: ${err.message}</p>`;
//         });
// }

// // =========================
// // Image Preview
// // =========================
// function previewImage(event) {
//     const preview = document.getElementById("preview");
//     preview.src = URL.createObjectURL(event.target.files[0]);
// }

// // =========================
// // GSAP Animations
// // =========================
// window.addEventListener("load", () => {
//     gsap.from(".gsap-input", { opacity: 0, y: 20, duration: 0.8, stagger: 0.1 });
//     gsap.from(".gsap-slide-next", { opacity: 0, x: -20, duration: 0.5 });
//     gsap.from(".gsap-zoom-in", { scale: 0.8, opacity: 0, duration: 1 });
// });





// =========================
// Symptom Categories
// =========================
const symptoms = {
    general: ["Fever", "Fatigue", "Headache", "Body Pain", "Dizziness"],
    respiratory: ["Cough", "Cold", "Sore Throat", "Shortness of Breath", "Chest Pain"],
    digestive: ["Nausea", "Vomiting", "Diarrhea", "Stomach Pain", "Loss of Appetite"],
    skin: ["Rash", "Itching", "Redness", "Swelling"]
};

// =========================
// Populate Categories
// =========================
document.addEventListener("DOMContentLoaded", () => {
    const categorySelect = document.getElementById("category");
    if (categorySelect) {
        Object.keys(symptoms).forEach(cat => {
            const option = document.createElement("option");
            option.value = cat;
            option.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
            categorySelect.appendChild(option);
        });
    }

    setupQuestionnaireSteps();
});

// =========================
// Show Symptoms
// =========================
function showSymptoms() {
    const category = document.getElementById("category").value;
    const container = document.getElementById("symptoms-list");
    container.innerHTML = "";

    if (category && symptoms[category]) {
        symptoms[category].forEach((symptom, index) => {
            const wrapper = document.createElement("div");
            wrapper.classList.add("form-check", "symptom-item");
            wrapper.style.opacity = "0";

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.classList.add("form-check-input");
            checkbox.name = "symptoms";
            checkbox.value = symptom;
            checkbox.id = `symptom-${index}`;

            const label = document.createElement("label");
            label.classList.add("form-check-label");
            label.textContent = symptom;
            label.setAttribute("for", `symptom-${index}`);
            label.style.cursor = "pointer";

            label.addEventListener("click", () => {
                checkbox.checked = !checkbox.checked;
            });

            wrapper.appendChild(checkbox);
            wrapper.appendChild(label);
            container.appendChild(wrapper);

            gsap.to(wrapper, {
                opacity: 1,
                y: 0,
                duration: 0.4,
                delay: index * 0.1,
                ease: "power2.out"
            });
        });
    }
}

// =========================
// Manual Symptom Recommendation
// =========================
function getSymptomRecommendation() {
    const checkedSymptoms = Array.from(document.querySelectorAll("input[name='symptoms']:checked"))
        .map(symptom => symptom.value);

    if (checkedSymptoms.length === 0) {
        alert("⚠️ Please select at least one symptom.");
        return;
    }

    fetch("/get_symptom_recommendation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: checkedSymptoms }),
    })
        .then(res => res.json())
        .then(data => {
            const resultDiv = document.getElementById("symptom-result");
            resultDiv.classList.remove("d-none", "alert-danger");
            resultDiv.classList.add("alert-info");

            if (data.error) {
                resultDiv.classList.add("alert-danger");
                resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
            } else {
                const medsList = Array.isArray(data.recommendation)
                    ? data.recommendation.join(", ")
                    : data.recommendation;
                resultDiv.innerHTML = `
                    <h5><strong>Result:</strong></h5>
                    <p><strong>Possible Cause:</strong> ${data.possible_cause}</p>
                    <p><strong>Recommended Medicine:</strong> ${medsList}</p>
                `;
            }
        })
        .catch(err => alert("Error: " + err));
}

// Mirror button
function getPossibleCause() {
    getSymptomRecommendation();
}

// =========================
// Questionnaire Steps (Fixed)
// =========================
function setupQuestionnaireSteps() {
    const steps = document.querySelectorAll(".step");
    const nextBtns = document.querySelectorAll(".gsap-slide-next");
    const progressBar = document.querySelector(".progress-bar");
    let currentStep = 0;

    function showStep(index) {
        steps.forEach((step, i) => {
            step.classList.toggle("active", i === index);
        });
        const progressPercent = ((index + 1) / steps.length) * 100;
        progressBar.style.width = `${progressPercent}%`;
        gsap.fromTo(".step.active", { opacity: 0, x: -40 }, { opacity: 1, x: 0, duration: 0.4 });
    }

    nextBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            if (currentStep < steps.length - 1) {
                currentStep++;
                showStep(currentStep);
            }
        });
    });

    showStep(currentStep);
}

// =========================
// Questionnaire Submit
// =========================
function getQuestionnaireRecommendation() {
    const data = {
        age: document.getElementById("age").value,
        gender: document.getElementById("gender").value,
        symptoms: document.getElementById("symptoms").value,
        duration: document.getElementById("duration").value,
        previous_treatment: document.getElementById("previous_treatment").value,
        taking_medicine: document.getElementById("Taking_Medicine").value
    };

    fetch("/get_questionnaire_recommendation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    })
        .then(res => res.json())
        .then(data => {
            const resultDiv = document.getElementById("questionnaire-result");
            resultDiv.classList.remove("d-none", "alert-danger");
            resultDiv.classList.add("alert-info");

            if (data.error) {
                resultDiv.classList.add("alert-danger");
                resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
            } else {
                const meds = Array.isArray(data.recommendation)
                    ? data.recommendation.join(", ")
                    : data.recommendation;
                resultDiv.innerHTML = `
                    <h5><strong>Disease:</strong> ${data.disease}</h5>
                    <p><strong>Possible Cause:</strong> ${data.possible_cause}</p>
                    <p><strong>Recommended Medicines:</strong> ${meds}</p>
                `;
            }
        })
        .catch(err => alert("Error: " + err));
}

// =========================
// Image Upload & Predict (Fixed)
// =========================
function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
        const preview = document.getElementById("preview");
        preview.src = reader.result;
    };
    reader.readAsDataURL(event.target.files[0]);
}

function uploadImageAndPredict(modelType = "TONGUE") {
    const fileInput = document.getElementById("image");
    if (!fileInput.files.length) {
        alert("Please upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    formData.append("model_type", modelType);

    fetch("/upload_and_predict", {
        method: "POST",
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            const resultDiv = document.getElementById("image-analysis-result");
            resultDiv.classList.remove("d-none", "alert-danger");
            resultDiv.classList.add("alert-info");

            if (data.error) {
                resultDiv.classList.add("alert-danger");
                resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
            } else {
                resultDiv.innerHTML = `
                    <h5><strong>Image Analysis Result</strong></h5>
                    <p><strong>Possible Cause:</strong> ${data.possible_cause}</p>
                    <p><strong>Recommendation:</strong> ${data.recommendation.join(", ")}</p>
                    <img src="${data.path}" class="img-fluid rounded mt-2" style="max-width:150px;">
                `;
            }
        })
        .catch(err => alert("Error: " + err));
}



// =========================
// GSAP Animations
// =========================
window.addEventListener("load", () => {
    gsap.from(".gsap-input", { opacity: 0, y: 20, duration: 0.8, stagger: 0.1 });
});

