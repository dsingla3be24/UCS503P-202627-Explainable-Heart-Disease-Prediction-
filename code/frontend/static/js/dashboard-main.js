        // SciFi Audio Controller using Web Audio API to synthesize sounds without files
        const SciFiAudioController = {
            ctx: null,
            scanOsc: null,
            scanGain: null,
            sonarTimer: null,

            init() {
                if (!this.ctx) {
                    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                    if (AudioContextClass) {
                        this.ctx = new AudioContextClass();
                    }
                }
                if (this.ctx && this.ctx.state === 'suspended') {
                    this.ctx.resume();
                }
            },

            startScanningSound() {
                this.init();
                if (!this.ctx) return;

                try {
                    this.scanOsc = this.ctx.createOscillator();
                    this.scanOsc.type = 'sawtooth';
                    this.scanOsc.frequency.setValueAtTime(60, this.ctx.currentTime);

                    const filter = this.ctx.createBiquadFilter();
                    filter.type = 'lowpass';
                    filter.frequency.setValueAtTime(150, this.ctx.currentTime);

                    this.scanGain = this.ctx.createGain();
                    this.scanGain.gain.setValueAtTime(0.0, this.ctx.currentTime);
                    this.scanGain.gain.linearRampToValueAtTime(0.12, this.ctx.currentTime + 0.3);

                    this.scanOsc.connect(filter);
                    filter.connect(this.scanGain);
                    this.scanGain.connect(this.ctx.destination);

                    this.scanOsc.start();

                    this.sonarTimer = setInterval(() => {
                        this.playSonarSweep();
                    }, 1200);
                } catch (e) {
                    console.error("Failed to start scan sound:", e);
                }
            },

            stopScanningSound() {
                if (this.sonarTimer) {
                    clearInterval(this.sonarTimer);
                    this.sonarTimer = null;
                }

                if (this.scanGain && this.ctx) {
                    try {
                        const curTime = this.ctx.currentTime;
                        this.scanGain.gain.cancelScheduledValues(curTime);
                        this.scanGain.gain.setValueAtTime(this.scanGain.gain.value, curTime);
                        this.scanGain.gain.linearRampToValueAtTime(0.0, curTime + 0.2);

                        const osc = this.scanOsc;
                        setTimeout(() => {
                            try { osc.stop(); } catch (e) { }
                        }, 250);
                    } catch (e) { }
                }
                this.scanOsc = null;
                this.scanGain = null;
            },

            playSonarSweep() {
                if (!this.ctx) return;
                try {
                    const osc = this.ctx.createOscillator();
                    const gain = this.ctx.createGain();

                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(800, this.ctx.currentTime);
                    osc.frequency.exponentialRampToValueAtTime(150, this.ctx.currentTime + 0.8);

                    gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.8);

                    osc.connect(gain);
                    gain.connect(this.ctx.destination);

                    osc.start();
                    osc.stop(this.ctx.currentTime + 0.85);
                } catch (e) { }
            },

            playTick() {
                this.init();
                if (!this.ctx) return;
                try {
                    const osc = this.ctx.createOscillator();
                    const gain = this.ctx.createGain();

                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(1400 + Math.random() * 600, this.ctx.currentTime);

                    gain.gain.setValueAtTime(0.015, this.ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.05);

                    osc.connect(gain);
                    gain.connect(this.ctx.destination);

                    osc.start();
                    osc.stop(this.ctx.currentTime + 0.06);
                } catch (e) { }
            },

            playSuccessChime() {
                this.init();
                if (!this.ctx) return;
                try {
                    const notes = [523.25, 659.25, 783.99, 1046.50];
                    const now = this.ctx.currentTime;

                    notes.forEach((freq, idx) => {
                        const osc = this.ctx.createOscillator();
                        const gain = this.ctx.createGain();

                        osc.type = 'triangle';
                        osc.frequency.setValueAtTime(freq, now + idx * 0.1);

                        gain.gain.setValueAtTime(0.0, now + idx * 0.1);
                        gain.gain.linearRampToValueAtTime(0.08, now + idx * 0.1 + 0.03);
                        gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.1 + 0.6);

                        const filter = this.ctx.createBiquadFilter();
                        filter.type = 'bandpass';
                        filter.frequency.setValueAtTime(freq * 1.5, now + idx * 0.1);
                        filter.Q.setValueAtTime(1.5, now + idx * 0.1);

                        osc.connect(filter);
                        filter.connect(gain);
                        gain.connect(this.ctx.destination);

                        osc.start(now + idx * 0.1);
                        osc.stop(now + idx * 0.1 + 0.65);
                    });
                } catch (e) { }
            },

            playInputTick() {
                this.init();
                if (!this.ctx) return;
                try {
                    const osc = this.ctx.createOscillator();
                    const gain = this.ctx.createGain();

                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(600 + Math.random() * 200, this.ctx.currentTime);

                    gain.gain.setValueAtTime(0.012, this.ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.03);

                    osc.connect(gain);
                    gain.connect(this.ctx.destination);

                    osc.start();
                    osc.stop(this.ctx.currentTime + 0.04);
                } catch (e) { }
            },

            playButtonClick() {
                this.init();
                if (!this.ctx) return;
                try {
                    const now = this.ctx.currentTime;
                    const osc1 = this.ctx.createOscillator();
                    const gain1 = this.ctx.createGain();
                    osc1.type = 'sine';
                    osc1.frequency.setValueAtTime(500, now);
                    gain1.gain.setValueAtTime(0.015, now);
                    gain1.gain.exponentialRampToValueAtTime(0.0001, now + 0.05);
                    osc1.connect(gain1);
                    gain1.connect(this.ctx.destination);
                    osc1.start(now);
                    osc1.stop(now + 0.06);

                    const osc2 = this.ctx.createOscillator();
                    const gain2 = this.ctx.createGain();
                    osc2.type = 'sine';
                    osc2.frequency.setValueAtTime(800, now + 0.04);
                    gain2.gain.setValueAtTime(0.015, now + 0.04);
                    gain2.gain.exponentialRampToValueAtTime(0.0001, now + 0.09);
                    osc2.connect(gain2);
                    gain2.connect(this.ctx.destination);
                    osc2.start(now + 0.04);
                    osc2.stop(now + 0.1);
                } catch (e) { }
            }
        };

        // Initialize 3D Vanta background
        window.vantaEffect = VANTA.NET({
            el: "#vanta-canvas",
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200.00,
            minWidth: 200.00,
            scale: 1.00,
            scaleMobile: 1.00,
            color: 0x38bdf8,
            backgroundColor: 0x050811,
            points: 12.00,
            maxDistance: 22.00,
            spacing: 16.00
        });





        // Dynamic Clinical Ranges Reference updates and Glowing Value Badges classes
        function updateReferences() {
            // Age Reference & Glowing Badge
            const ageVal = parseInt(document.getElementById('age').value);
            const ageRef = document.getElementById('age-ref');
            const ageBadge = document.getElementById('val-age');
            if (ageVal < 45) {
                ageRef.className = 'text-success';
                ageRef.innerHTML = '&bull; Young Adult';
                if (ageBadge) ageBadge.className = 'glowing-badge normal';
            } else if (ageVal < 65) {
                ageRef.className = 'text-info';
                ageRef.innerHTML = '&bull; Middle-Age';
                if (ageBadge) ageBadge.className = 'glowing-badge warning';
            } else {
                ageRef.className = 'text-warning';
                ageRef.innerHTML = '&bull; Senior';
                if (ageBadge) ageBadge.className = 'glowing-badge danger';
            }

            // BP Reference & Glowing Badge
            const bpVal = parseInt(document.getElementById('trestbps').value);
            const bpRef = document.getElementById('bp-ref');
            const bpBadge = document.getElementById('val-trestbps');
            if (bpVal <= 120) {
                bpRef.className = 'text-success';
                bpRef.innerHTML = '&bull; Healthy';
                if (bpBadge) bpBadge.className = 'glowing-badge normal';
            } else if (bpVal <= 139) {
                bpRef.className = 'text-warning';
                bpRef.innerHTML = '&bull; Prehypertension';
                if (bpBadge) bpBadge.className = 'glowing-badge warning';
            } else {
                bpRef.className = 'text-danger';
                bpRef.innerHTML = '&bull; Hypertension';
                if (bpBadge) bpBadge.className = 'glowing-badge danger';
            }

            // Cholesterol Reference & Glowing Badge
            const cholVal = parseInt(document.getElementById('chol').value);
            const cholRef = document.getElementById('chol-ref');
            const cholBadge = document.getElementById('val-chol');
            if (cholVal < 200) {
                cholRef.className = 'text-success';
                cholRef.innerHTML = '&bull; Desirable';
                if (cholBadge) cholBadge.className = 'glowing-badge normal';
            } else if (cholVal <= 239) {
                cholRef.className = 'text-warning';
                cholRef.innerHTML = '&bull; Borderline High';
                if (cholBadge) cholBadge.className = 'glowing-badge warning';
            } else {
                cholRef.className = 'text-danger';
                cholRef.innerHTML = '&bull; High Risk';
                if (cholBadge) cholBadge.className = 'glowing-badge danger';
            }

            // Max Heart Rate Reference & Glowing Badge
            const hrVal = parseInt(document.getElementById('thalach').value);
            const hrRef = document.getElementById('hr-ref');
            const hrBadge = document.getElementById('val-thalach');
            const age = parseInt(document.getElementById('age').value);
            const predictedMax = 220 - age;
            const hrPercent = hrVal / predictedMax;
            if (hrPercent > 0.85) {
                hrRef.className = 'text-success';
                hrRef.innerHTML = '&bull; High Capability';
                if (hrBadge) hrBadge.className = 'glowing-badge normal';
            } else if (hrPercent >= 0.6) {
                hrRef.className = 'text-info';
                hrRef.innerHTML = '&bull; Normal';
                if (hrBadge) hrBadge.className = 'glowing-badge warning';
            } else {
                hrRef.className = 'text-warning';
                hrRef.innerHTML = '&bull; Low Response';
                if (hrBadge) hrBadge.className = 'glowing-badge danger';
            }

            // ST Depression Reference & Glowing Badge
            const oldpeakVal = parseFloat(document.getElementById('oldpeak').value);
            const oldpeakRef = document.getElementById('oldpeak-ref');
            const oldpeakBadge = document.getElementById('val-oldpeak');
            if (oldpeakVal <= 0.5) {
                oldpeakRef.className = 'text-success';
                oldpeakRef.innerHTML = '&bull; Normal';
                if (oldpeakBadge) oldpeakBadge.className = 'glowing-badge normal';
            } else if (oldpeakVal < 2.0) {
                oldpeakRef.className = 'text-warning';
                oldpeakRef.innerHTML = '&bull; Mild Depression';
                if (oldpeakBadge) oldpeakBadge.className = 'glowing-badge warning';
            } else {
                oldpeakRef.className = 'text-danger';
                oldpeakRef.innerHTML = '&bull; Severe Depression';
                if (oldpeakBadge) oldpeakBadge.className = 'glowing-badge danger';
            }
        }

        // Initialize Clinical Reference text labels
        updateReferences();

        // Dynamic Label / Slider synchronization
        const syncInputs = [
            { id: 'age', displayId: 'val-age' },
            { id: 'trestbps', displayId: 'val-trestbps' },
            { id: 'chol', displayId: 'val-chol' },
            { id: 'thalach', displayId: 'val-thalach' },
            { id: 'oldpeak', displayId: 'val-oldpeak', isFloat: true }
        ];

        syncInputs.forEach(item => {
            const inputEl = document.getElementById(item.id);
            const displayEl = document.getElementById(item.displayId);

            inputEl.addEventListener('input', (e) => {
                const val = e.target.value;
                displayEl.value = item.isFloat ? parseFloat(val).toFixed(1) : val;
                updateReferences();
            });

            // Reverse sync: User typing in the input box updates the slider
            displayEl.addEventListener('input', (e) => {
                const val = e.target.value;
                inputEl.value = val;
                updateReferences();
            });
        });

        window.enableDemographics = function() {
            const ageInput = document.getElementById('age');
            const ageValInput = document.getElementById('val-age');
            const sexSelect = document.getElementById('sex');
            const patientIdInput = document.getElementById('patient_id');
            ageInput.disabled = false;
            ageValInput.disabled = false;
            sexSelect.disabled = false;
            patientIdInput.value = ''; // clear when switching to new
        };

        window.autocompletePatientsData = [];
        
        window.disableDemographics = function() {
            const ageInput = document.getElementById('age');
            const ageValInput = document.getElementById('val-age');
            const sexSelect = document.getElementById('sex');
            ageInput.disabled = true;
            ageValInput.disabled = true;
            sexSelect.disabled = true;
            
            // Fetch datalist options for autocomplete
            // Frontend Prototype: Mock empty patient list (no backend)
            Promise.resolve({ json: () => Promise.resolve([]) })
                .then(res => res.json())
                .then(data => {
                    if (data && data.length > 0) {
                        window.autocompletePatientsData = data;
                        const nameList = document.getElementById('patient_name_list');
                        const idList = document.getElementById('patient_id_list');
                        nameList.innerHTML = '';
                        idList.innerHTML = '';
                        data.forEach(p => {
                            nameList.innerHTML += `<option value="${p.name}">`;
                            idList.innerHTML += `<option value="${p.hospital_patient_id}">`;
                        });
                    }
                })
                .catch(err => console.error("Error loading autocomplete data:", err));
        };

        const patientNameField = document.getElementById('patient_name');
        if (patientNameField) {
            patientNameField.addEventListener('change', (e) => {
                if (window.autocompletePatientsData) {
                    const match = window.autocompletePatientsData.find(p => p.name === e.target.value.trim());
                    if (match && document.getElementById('patient_id')) {
                        document.getElementById('patient_id').value = match.hospital_patient_id;
                        // trigger blur manually to load demographics
                        document.getElementById('patient_id').dispatchEvent(new Event('blur'));
                    }
                }
            });
        }

        const patientIdField = document.getElementById('patient_id');
        if (patientIdField) {
            patientIdField.addEventListener('change', (e) => {
                if (window.autocompletePatientsData) {
                    const match = window.autocompletePatientsData.find(p => p.hospital_patient_id === e.target.value.trim());
                    if (match && document.getElementById('patient_name')) {
                        document.getElementById('patient_name').value = match.name;
                    }
                }
            });
            patientIdField.addEventListener('blur', async (e) => {
                const query = e.target.value.trim();
                if (!query) return;
                
                try {
                    // Frontend Prototype: Mock empty search (no backend)
                    const data = [];
                    
                    if (data && data.length > 0) {
                        const patient = data[0];
                        if (patient.history && patient.history.length > 0) {
                            const latest = patient.history.sort((a, b) => new Date(b.recorded_at) - new Date(a.recorded_at))[0];
                            
                            const msDiff = new Date() - new Date(latest.recorded_at);
                            const yearsDiff = msDiff / (1000 * 60 * 60 * 24 * 365.25);
                            const newAge = Math.floor(latest.age + yearsDiff);
                            
                            document.getElementById('age').value = newAge;
                            document.getElementById('val-age').value = newAge;
                            
                            // API returns 'Male' or 'Female' string
                            const sexVal = (latest.sex && latest.sex.toLowerCase() === 'female') ? "1" : "0";
                            document.getElementById('sex').value = sexVal;
                            
                            updateReferences();
                        }
                    }
                } catch (err) {
                    console.error("Error fetching patient demographics:", err);
                }
            });
        }



        // Form Submit -> Diagnosis trigger
        document.getElementById('diagnosis-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const btnDiagnose = document.getElementById('btn-diagnose');
            const dashboard = document.getElementById('results-dashboard');
            const scanner = document.getElementById('scanner-laser');
            const resultsScanner = document.getElementById('results-scanner');

            // Progress elements
            const progressBar = document.getElementById('scanner-progress-bar');
            const progressText = document.getElementById('scanner-progress-text');
            if (progressBar) progressBar.style.width = "0%";
            if (progressText) progressText.textContent = "System Initializing... (0%)";

            btnDiagnose.disabled = true;
            btnDiagnose.textContent = "COMPUTING AI DIAGNOSIS...";

            // Start parameters scanning visual effect
            if (scanner) scanner.classList.add('scanning');

            // Play continuous scanner audio loop
            SciFiAudioController.startScanningSound();

            // Hide dashboard and reveal robotic scan screen
            dashboard.classList.add('d-none');
            if (resultsScanner) {
                resultsScanner.classList.remove('d-none');
                resultsScanner.classList.add('fullscreen-overlay');
            }

            // Gather form parameters
            const formData = new FormData(e.target);
            const rawData = {};
            const patientType = formData.get('patient_type');
            
            formData.forEach((val, key) => {
                if (key === 'patient_name' || key === 'patient_id' || key === 'patient_type') {
                    if (key === 'patient_id' && patientType === 'new') return; // ignore patient_id for new patients
                    rawData[key] = val;
                } else {
                    rawData[key] = parseFloat(val);
                }
            });

            try {
                // ── Frontend Prototype: Mock Demo Response ──
                // Generates a realistic demo prediction based on user inputs
                // (Will be replaced with actual backend API call during integration)
                const mockProbability = Math.min(0.98, Math.max(0.05,
                    0.15 +
                    (rawData.age > 55 ? 0.12 : 0) +
                    (rawData.cp == 4 ? 0.15 : rawData.cp == 3 ? 0.05 : 0) +
                    (rawData.chol > 240 ? 0.12 : 0) +
                    (rawData.trestbps > 140 ? 0.1 : 0) +
                    (rawData.oldpeak > 2 ? 0.1 : 0) +
                    (rawData.ca > 0 ? 0.08 * rawData.ca : 0) +
                    (rawData.thal == 7 ? 0.12 : rawData.thal == 6 ? 0.08 : 0) +
                    (rawData.exang == 1 ? 0.08 : 0) +
                    (rawData.thalach < 120 ? 0.08 : -0.05) +
                    (Math.random() * 0.06 - 0.03)
                ));
                const featureNames = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];
                const mockAttention = featureNames.map(() => +(Math.random() * 0.3 + 0.02).toFixed(4));
                const mockGradcam = featureNames.map(() => +(Math.random() * 0.4 + 0.01).toFixed(4));
                const mockShap = featureNames.map(() => +(Math.random() * 0.25 - 0.05).toFixed(4));
                const gradcamMatrix = featureNames.map(() => featureNames.map(() => +(Math.random() * 0.5).toFixed(3)));
                const results = {
                    probability: mockProbability,
                    true_target: mockProbability > 0.5 ? 1 : 0,
                    patient_name: rawData.patient_name || 'Demo Patient',
                    hospital_patient_id: 'PT-DEMO-' + Math.random().toString(36).substring(2, 8).toUpperCase(),
                    model_metrics: { accuracy: 0.9344, precision: 0.9412, recall: 0.9275, f1: 0.9343, auc: 0.9780 },
                    attention_weights: mockAttention,
                    gradcam_weights: mockGradcam,
                    shap_values: mockShap,
                    feature_names: featureNames,
                    gradcam_2d_matrix: gradcamMatrix,
                    gradcam_2d_labels: featureNames,
                    primary_drivers: [
                        { feature: 'Chest Pain Type', value: rawData.cp == 4 ? 'Asymptomatic' : 'Anginal', risk: rawData.cp == 4 ? 'HIGH' : 'LOW', explanation: 'Asymptomatic chest pain is the most dangerous presentation.' },
                        { feature: 'Max Heart Rate', value: rawData.thalach + ' bpm', risk: rawData.thalach < 120 ? 'HIGH' : 'NORMAL', explanation: 'Chronotropic incompetence suggests reduced cardiac output.' },
                        { feature: 'ST Depression', value: rawData.oldpeak, risk: rawData.oldpeak > 2 ? 'HIGH' : 'MODERATE', explanation: 'ST segment depression indicates subendocardial ischemia.' }
                    ],
                    secondary_observations: [
                        { feature: 'Cholesterol', value: rawData.chol + ' mg/dl', risk: rawData.chol > 240 ? 'ELEVATED' : 'NORMAL', explanation: 'Elevated serum cholesterol accelerates arterial plaque buildup.' },
                        { feature: 'Resting BP', value: rawData.trestbps + ' mmHg', risk: rawData.trestbps > 140 ? 'HIGH' : 'NORMAL', explanation: 'Hypertension is a chronic cardiovascular strain indicator.' }
                    ],
                    narrative: [
                        `Patient ${rawData.patient_name || 'Demo'}, age ${rawData.age}, presents with ${rawData.cp == 4 ? 'asymptomatic' : 'anginal'} chest pain profile.`,
                        `Resting BP of ${rawData.trestbps} mmHg and cholesterol at ${rawData.chol} mg/dl ${rawData.chol > 240 ? 'exceed recommended thresholds' : 'are within acceptable range'}.`,
                        `Model consensus probability: ${(mockProbability * 100).toFixed(1)}%. ${mockProbability > 0.5 ? 'Clinical follow-up is strongly advised.' : 'No immediate intervention required.'}`
                    ],
                    pdf_url: '#'
                };

                // Log templates for cascade animation
                const logTemplates = [
                    "[SYSTEM] Initiating physiological data ingestion...",
                    "[SYSTEM] Mapping age, sex, and chest pain parameters...",
                    "[Vitals] Validating blood pressure and cholesterol bounds...",
                    "[ECG] Synthesizing ST-waveform metrics...",
                    "[AI] Converting 1D telemetry array to 2D Gramian Field (GASF)...",
                    "[AI] Feeding spatial dimensions to CNN-LSTM-Attention layers...",
                    "[XAI] Generating Grad-CAM feature activations...",
                    "[XAI] Executing SHAP KernelExplainer local attributions...",
                    "[MODEL] Decision consensus: SUCCESS.",
                    "[SYSTEM] Loading visualization telemetry dashboards..."
                ];

                // Cascade scrolling log terminal logic
                const termLines = [
                    document.getElementById('term-line-1'),
                    document.getElementById('term-line-2'),
                    document.getElementById('term-line-3'),
                    document.getElementById('term-line-4'),
                    document.getElementById('term-line-5')
                ];

                // Clear terminal lines
                termLines.forEach(lineEl => { if (lineEl) lineEl.textContent = ""; });

                let lineIdx = 0;
                let printCount = 0;

                function printNextLine() {
                    if (printCount >= logTemplates.length) {
                        if (progressBar) progressBar.style.width = "100%";
                        if (progressText) progressText.textContent = "ANALYSIS COMPLETE (100%)";

                        // Stop scanner hum and play success arpeggio chime
                        SciFiAudioController.stopScanningSound();
                        SciFiAudioController.playSuccessChime();

                        setTimeout(() => {
                            // Hide parameter scanning visual effect
                            if (scanner) scanner.classList.remove('scanning');

                            if (resultsScanner) {
                                resultsScanner.classList.add('fade-out');
                                setTimeout(() => {
                                    resultsScanner.classList.add('d-none');
                                    resultsScanner.classList.remove('fullscreen-overlay', 'fade-out');

                                    // Reset dashboard animations to re-trigger stagger entry effects
                                    dashboard.querySelectorAll('.fade-in-up').forEach(el => {
                                        el.style.animation = 'none';
                                        el.offsetHeight; // Trigger reflow to restart animation
                                        el.style.animation = '';
                                    });

                                    // Hide control panel and show results column
                                    const ctrlCol = document.getElementById('control-panel-column');
                                    if (ctrlCol) ctrlCol.classList.add('d-none');
                                    const resCol = document.getElementById('results-column');
                                    if (resCol) {
                                        resCol.classList.remove('d-none');
                                    }

                                    // Display Dashboard
                                    dashboard.classList.remove('d-none');

                                    // Render Plotly graphs and PDF elements
                                    finishLoadingResults(results);
                                }, 500);
                            } else {
                                // Hide control panel and show results column
                                const ctrlCol = document.getElementById('control-panel-column');
                                if (ctrlCol) ctrlCol.classList.add('d-none');
                                const resCol = document.getElementById('results-column');
                                if (resCol) {
                                    resCol.classList.remove('d-none');
                                }

                                dashboard.classList.remove('d-none');
                                finishLoadingResults(results);
                            }
                        }, 400);
                        return;
                    }

                    const timestamp = new Date().toISOString().split('T')[1].substring(0, 8);
                    const lineContent = `[${timestamp}] ${logTemplates[printCount]}`;

                    if (lineIdx < 5) {
                        if (termLines[lineIdx]) termLines[lineIdx].textContent = lineContent;
                        lineIdx++;
                    } else {
                        for (let i = 0; i < 4; i++) {
                            if (termLines[i] && termLines[i + 1]) {
                                termLines[i].textContent = termLines[i + 1].textContent;
                            }
                        }
                        if (termLines[4]) termLines[4].textContent = lineContent;
                    }

                    // Play typewriter data tick sound
                    SciFiAudioController.playTick();

                    // Increment progress bar and status message
                    const percent = Math.round(((printCount + 1) / logTemplates.length) * 100);
                    if (progressBar) progressBar.style.width = `${percent}%`;
                    if (progressText) {
                        let statusMsg = "Processing";
                        if (percent < 30) statusMsg = "Ingesting Telemetry Data";
                        else if (percent < 60) statusMsg = "Synthesizing Waveforms";
                        else if (percent < 85) statusMsg = "Running CNN-LSTM-Attention Layers";
                        else statusMsg = "Compiling Explainability Attributions";
                        progressText.textContent = `${statusMsg}... (${percent}%)`;
                    }

                    printCount++;
                    const delay = 150 + Math.random() * 150;
                    setTimeout(printNextLine, delay);
                }

                printNextLine();

                // Separate function to render Plotly graphs and text metrics
                function finishLoadingResults(results) {
                    // 1. Update Diagnosis Card
                    const prob = results.probability;
                    const targetProb = prob * 100;

                    const statusBadge = document.getElementById('display-status-badge');
                    const displayHeart = document.getElementById('display-heart');
                    const riskMsg = document.getElementById('display-risk-msg');

                    const nameBadge = document.getElementById('display-patient-name');
                    if (nameBadge) {
                        nameBadge.textContent = results.patient_name ? `- ${results.patient_name}` : "";
                    }

                    const idBadge = document.getElementById('display-patient-id');
                    if (idBadge) {
                        idBadge.textContent = results.hospital_patient_id ? `[${results.hospital_patient_id}]` : "";
                    }

                    if (prob >= 0.5) {
                        statusBadge.textContent = "HIGH RISK";
                        statusBadge.className = "status-badge danger";
                        displayHeart.className = "beating-heart text-danger";
                        riskMsg.textContent = "Elevated cardiovascular risk markers. Clinical follow-up highly recommended.";
                    } else {
                        statusBadge.textContent = "LOW RISK / NORMAL";
                        statusBadge.className = "status-badge success";
                        displayHeart.className = "beating-heart text-success normal";
                        riskMsg.textContent = "No significant risk signals identified. Cardiovascular vitals within normal parameters.";
                    }

                    // Apply dynamic heart rate speed based on probability
                    const pulseDuration = prob >= 0.5
                        ? Math.max(0.4, 1.4 - prob).toFixed(2)
                        : Math.max(1.0, 1.8 - prob * 1.5).toFixed(2);
                    displayHeart.style.animationDuration = `${pulseDuration}s`;

                    // Show Ground Truth target if matched and auto-detected by backend
                    const trueTargetVal = results.true_target;
                    const gtEl = document.getElementById('display-ground-truth');
                    if (trueTargetVal !== null && trueTargetVal !== undefined) {
                        const gtLabel = parseInt(trueTargetVal) === 1 ? "Heart Disease" : "Normal";
                        gtEl.textContent = `Ground Truth Dataset Label: ${gtLabel}`;
                        gtEl.style.display = "block";
                    } else {
                        gtEl.style.display = "none";
                    }

                    // Animate probability value counting up
                    const pBar = document.getElementById('display-progress-bar');
                    const pText = document.getElementById('display-probability');

                    let currentProb = 0;
                    const duration = 1200; // 1.2 seconds
                    const startTime = performance.now();

                    if (prob >= 0.5) {
                        pBar.className = "progress-bar progress-bar-striped progress-bar-animated bg-danger";
                    } else {
                        pBar.className = "progress-bar progress-bar-striped progress-bar-animated bg-success";
                    }

                    function animateProbability(timestamp) {
                        const elapsed = timestamp - startTime;
                        const progress = Math.min(elapsed / duration, 1);

                        const easeProgress = progress * (2 - progress);
                        currentProb = easeProgress * targetProb;

                        if (pText) pText.textContent = currentProb.toFixed(1) + "%";
                        if (pBar) pBar.style.width = currentProb + "%";

                        if (progress < 1) {
                            requestAnimationFrame(animateProbability);
                        } else {
                            if (pText) pText.textContent = targetProb.toFixed(1) + "%";
                            if (pBar) pBar.style.width = targetProb + "%";
                        }
                    }
                    requestAnimationFrame(animateProbability);

                    // Update Vanta background parameters dynamically
                    if (window.vantaEffect) {
                        if (prob >= 0.5) {
                            window.vantaEffect.setOptions({
                                color: 0xf43f5e,
                                points: 15.00,
                                spacing: 14.00
                            });
                        } else {
                            window.vantaEffect.setOptions({
                                color: 0x38bdf8,
                                points: 12.00,
                                spacing: 16.00
                            });
                        }
                    }



                    // 3. Render PDF Download Button
                    const btnPdf = document.getElementById('btn-download-pdf');
                    if (results.pdf_url) {
                        btnPdf.href = results.pdf_url;
                        btnPdf.classList.remove('disabled');
                    } else {
                        btnPdf.classList.add('disabled');
                    }

                    // 4. Render XAI Multi-Bar Chart
                    renderPlotlyAttributions(results.xai);

                    // 6. Render 2D Grad-CAM Heatmap
                    renderPlotlyGradCam(results.grad_cam_2d, results.xai.processed_features);

                    // 7. Populate Clinical Abnormal Flags with Animations
                    populateAbnormalFlags(results.clinical_details);

                    // 8. Populate Model Performance Metrics
                    if (results.metrics) {
                        document.getElementById('metric-accuracy').textContent = (results.metrics.accuracy * 100).toFixed(1) + "%";
                        document.getElementById('metric-precision').textContent = (results.metrics.precision * 100).toFixed(1) + "%";
                        document.getElementById('metric-recall').textContent = (results.metrics.recall * 100).toFixed(1) + "%";
                        document.getElementById('metric-f1').textContent = (results.metrics.f1 * 100).toFixed(1) + "%";
                        document.getElementById('metric-auc').textContent = (results.metrics.roc_auc * 100).toFixed(1) + "%";

                        const metricsTitle = document.getElementById('metrics-title');
                        if (metricsTitle) {
                            metricsTitle.textContent = "Model Performance Metrics";
                        }
                    }
                }

            } catch (err) {
                // Stop scanning sound on error
                SciFiAudioController.stopScanningSound();
                console.error(err);
                alert("An error occurred during cardiac diagnosis: " + err.message);
            } finally {
                btnDiagnose.disabled = false;
                btnDiagnose.textContent = "RUN CARDIAC DIAGNOSIS";
                if (scanner) scanner.classList.remove('scanning');
            }
        });

        // Reset pipeline / Test another patient listener
        document.getElementById('btn-reset-pipeline').addEventListener('click', () => {
            // Restore left control panel column
            const ctrlCol = document.getElementById('control-panel-column');
            if (ctrlCol) ctrlCol.classList.remove('d-none');

            // Hide results column
            const resCol = document.getElementById('results-column');
            if (resCol) {
                resCol.classList.add('d-none');
            }

            // Hide dashboard results
            document.getElementById('results-dashboard').classList.add('d-none');



            // Reset the submit button state
            const btnDiagnose = document.getElementById('btn-diagnose');
            if (btnDiagnose) {
                btnDiagnose.disabled = false;
                btnDiagnose.textContent = "RUN CARDIAC DIAGNOSIS";
            }

            // Scroll smoothly back to top of parameter card
            const parameterCard = document.getElementById('parameter-card');
            if (parameterCard) {
                parameterCard.scrollIntoView({ behavior: 'smooth' });
            }
        });




        // Plotly attribution chart renderer
        function renderPlotlyAttributions(xai) {
            // Sort features by SHAP value for a cleaner look
            const zip = xai.features.map((f, i) => ({
                feat: f.toUpperCase(),
                att: xai.attention[i],
                gc: xai.grad_cam[i],
                sh: xai.shap[i]
            }));
            zip.sort((a, b) => b.sh - a.sh);

            const features = zip.map(z => z.feat);
            const attention = zip.map(z => z.att);
            const grad_cam = zip.map(z => z.gc);
            const shap = zip.map(z => z.sh);

            // Get current active theme color dynamically
            const themeColor = getComputedStyle(document.documentElement).getPropertyValue('--neon-blue').trim() || '#38bdf8';

            const traceShap = {
                x: features,
                y: shap,
                name: 'SHAP Contribution (Local)',
                type: 'bar',
                marker: { color: '#f43f5e', opacity: 0.85 }
            };

            const traceGradCam = {
                x: features,
                y: grad_cam,
                name: 'Grad-CAM Activation (Local)',
                type: 'bar',
                marker: { color: themeColor, opacity: 0.85 }
            };

            const traceAttention = {
                x: features,
                y: attention,
                name: 'Attention Weight (Global)',
                type: 'bar',
                marker: { color: '#818cf8', opacity: 0.85 }
            };

            const layout = {
                barmode: 'group',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e2e8f0', family: 'Plus Jakarta Sans' },
                xaxis: { tickangle: -45, tickfont: { size: 10 }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: 'Attribution strength (Normalized)', gridcolor: 'rgba(255,255,255,0.05)' },
                legend: { orientation: 'h', y: -0.6, x: 0.5, xanchor: 'center' },
                margin: { t: 20, b: 140, l: 50, r: 20 }
            };

            Plotly.newPlot('plotly-bar-chart', [traceShap, traceGradCam, traceAttention], layout, { responsive: true });
        }

        // Plotly 2D Grad-CAM Heatmap renderer
        function renderPlotlyGradCam(grad_cam_2d, features) {
            const cleanFeatures = features;

            const data = [{
                z: grad_cam_2d,
                x: cleanFeatures,
                y: cleanFeatures,
                type: 'heatmap',
                colorscale: 'Jet',
                showscale: true,
                colorbar: {
                    title: 'Influence Level',
                    titleside: 'right',
                    titlefont: { color: '#94a3b8' },
                    tickfont: { color: '#94a3b8' }
                }
            }];

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e2e8f0', family: 'Plus Jakarta Sans' },
                xaxis: { showgrid: false, zeroline: false, showticklabels: true, tickangle: -45, tickfont: { size: 9, color: '#94a3b8' } },
                yaxis: { showgrid: false, zeroline: false, showticklabels: true, tickfont: { size: 9, color: '#94a3b8' }, scaleanchor: "x" },
                margin: { t: 10, b: 70, l: 80, r: 10 }
            };

            Plotly.newPlot('plotly-gradcam-heatmap', data, layout, { responsive: true });

            // Interactive highlight feature logic
            const selectEl = document.getElementById('gradcam-feature-select');
            if (selectEl) {
                selectEl.innerHTML = '<option value="ALL">Show all interactions</option>';
                const uniqueFeatures = [...new Set(cleanFeatures)];
                uniqueFeatures.forEach(feat => {
                    const opt = document.createElement('option');
                    opt.value = feat;
                    opt.textContent = feat;
                    selectEl.appendChild(opt);
                });

                // Clone to remove old event listeners
                const newSelectEl = selectEl.cloneNode(true);
                selectEl.parentNode.replaceChild(newSelectEl, selectEl);

                newSelectEl.addEventListener('change', function(e) {
                    const selectedFeat = e.target.value;
                    let newZ;
                    
                    if (selectedFeat === 'ALL') {
                        newZ = grad_cam_2d; // Restore original heatmap
                    } else {
                        const matchingIndices = cleanFeatures.map((f, i) => f === selectedFeat ? i : -1).filter(i => i !== -1);
                        newZ = grad_cam_2d.map((row, r) => {
                            return row.map((val, c) => {
                                // Keep values for selected feature row/col, fade out the rest
                                if (matchingIndices.includes(r) || matchingIndices.includes(c)) return val;
                                return val * 0.15; 
                            });
                        });
                    }
                    
                    Plotly.restyle('plotly-gradcam-heatmap', { 'z': [newZ] });
                });
            }
        }



        // Typewriter text dynamic streaming animation supporting both string paragraphs and points lists
        function typewriterEffect(element, text, speed = 8) {
            element.innerHTML = "";
            if (element.typewriterTimer) {
                clearInterval(element.typewriterTimer);
            }

            if (Array.isArray(text)) {
                let ptIdx = 0;

                function typeNextPoint() {
                    if (ptIdx >= text.length) return;

                    const li = document.createElement('li');
                    li.style.marginBottom = "8px";
                    li.style.fontSize = "0.78rem";
                    element.appendChild(li);

                    const str = text[ptIdx];
                    let charIdx = 0;
                    let currentHtml = "";

                    element.typewriterTimer = setInterval(() => {
                        if (charIdx < str.length) {
                            if (str.charAt(charIdx) === '<') {
                                const closeIdx = str.indexOf('>', charIdx);
                                if (closeIdx !== -1) {
                                    currentHtml += str.substring(charIdx, closeIdx + 1);
                                    charIdx = closeIdx + 1;
                                } else {
                                    currentHtml += str.charAt(charIdx);
                                    charIdx++;
                                }
                            } else {
                                currentHtml += str.charAt(charIdx);
                                charIdx++;
                            }
                            li.innerHTML = currentHtml;
                        } else {
                            clearInterval(element.typewriterTimer);
                            ptIdx++;
                            typeNextPoint();
                        }
                    }, speed);
                }
                typeNextPoint();
            } else {
                let i = 0;
                element.typewriterTimer = setInterval(() => {
                    if (i < text.length) {
                        element.innerHTML += text.charAt(i);
                        i++;
                    } else {
                        clearInterval(element.typewriterTimer);
                    }
                }, speed);
            }
        }

        // Map clinical feature bounds to status colors and dynamic visual level gauges
        function getFeatureVisualProps(feature, val) {
            const featUpper = feature.toUpperCase();
            let iconSvg = "";
            let statusClass = "normal";
            let pct = 20; // default percentage for gauge

            // Define custom SVG paths for clinical indicators
            if (featUpper.includes("BP") || featUpper.includes("TRESTBPS")) {
                // Pressure cuff icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>`;
                const bp = parseFloat(val);
                if (bp >= 140) { statusClass = "danger"; pct = 85; }
                else if (bp >= 130) { statusClass = "warning"; pct = 60; }
                else { statusClass = "normal"; pct = 30; }
            }
            else if (featUpper.includes("CHOL")) {
                // Blood drop icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2L4.5 12A7.5 7.5 0 1 0 19.5 12L12 2Z"/></svg>`;
                const chol = parseFloat(val);
                if (chol >= 240) { statusClass = "danger"; pct = 85; }
                else if (chol >= 200) { statusClass = "warning"; pct = 55; }
                else { statusClass = "normal"; pct = 30; }
            }
            else if (featUpper.includes("THALACH")) {
                // Max HR - heart rate pulse icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>`;
                const hr = parseFloat(val);
                if (hr < 110) { statusClass = "danger"; pct = 25; }
                else if (hr < 140) { statusClass = "warning"; pct = 50; }
                else { statusClass = "normal"; pct = 80; }
            }
            else if (featUpper.includes("OLDPEAK")) {
                // ST depression - waveform segment scan icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M2 10h4l2-5 3 10 2-5h9"/></svg>`;
                const peak = parseFloat(val);
                if (peak >= 2.0) { statusClass = "danger"; pct = 85; }
                else if (peak >= 0.5) { statusClass = "warning"; pct = 50; }
                else { statusClass = "normal"; pct = 15; }
            }
            else if (featUpper.includes("AGE")) {
                // Profile icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>`;
                const age = parseFloat(val);
                if (age >= 65) { statusClass = "danger"; pct = 80; }
                else if (age >= 45) { statusClass = "warning"; pct = 55; }
                else { statusClass = "normal"; pct = 35; }
            }
            else if (featUpper.includes("SEX")) {
                // Gender icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="10" r="8"/><path d="M12 2v8M12 18H6M18 18h-6"/></svg>`;
                statusClass = parseFloat(val) === 1.0 ? "warning" : "normal";
                pct = parseFloat(val) === 1.0 ? 70 : 30;
            }
            else if (featUpper.includes("CP")) {
                // Chest pain type
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
                const cp = parseFloat(val);
                if (cp === 4.0) { statusClass = "danger"; pct = 90; }
                else if (cp === 3.0) { statusClass = "warning"; pct = 60; }
                else { statusClass = "normal"; pct = 30; }
            }
            else if (featUpper.includes("FBS") || featUpper.includes("SUGAR")) {
                // Fasting blood sugar - flask/chemical icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10 2v7.586l-5.707 5.707A2 2 0 0 0 6 19h12a2 2 0 0 0 1.707-3.707L14 9.586V2h-4z"/></svg>`;
                statusClass = parseFloat(val) === 1.0 ? "danger" : "normal";
                pct = parseFloat(val) === 1.0 ? 80 : 20;
            }
            else if (featUpper.includes("ECG") || featUpper.includes("RESTECG")) {
                // Resting ECG
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M3 3v18h18M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/></svg>`;
                const ecg = parseFloat(val);
                if (ecg > 0.0) { statusClass = "danger"; pct = 80; }
                else { statusClass = "normal"; pct = 20; }
            }
            else if (featUpper.includes("EXANG")) {
                // Exercise angina - activity running icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 8h4M6 8h4M2 8h4M12 2v20"/></svg>`;
                statusClass = parseFloat(val) === 1.0 ? "danger" : "normal";
                pct = parseFloat(val) === 1.0 ? 85 : 15;
            }
            else if (featUpper.includes("SLOPE")) {
                // ST slope - trending down icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M23 6l-9.5 9.5-5-5L1 18M17 6h6v6"/></svg>`;
                const slope = parseFloat(val);
                if (slope === 2.0 || slope === 3.0) { statusClass = "danger"; pct = 85; }
                else { statusClass = "normal"; pct = 20; }
            }
            else if (featUpper.includes("CA") || featUpper.includes("VESSEL")) {
                // Vessels blocked ca - anatomy circle node icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>`;
                const ca = parseFloat(val);
                if (ca > 0) { statusClass = "danger"; pct = 30 + ca * 20; }
                else { statusClass = "normal"; pct = 15; }
            }
            else if (featUpper.includes("THAL")) {
                // Thalassemia - crosshair blood scanner icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M12 2v4M12 18v4M4 12h4M16 12h4"/></svg>`;
                const thal = parseFloat(val);
                if (thal === 6.0 || thal === 7.0) { statusClass = "danger"; pct = 80; }
                else { statusClass = "normal"; pct = 20; }
            }
            else {
                // Generic shield icon
                iconSvg = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`;
            }

            return { iconSvg, statusClass, pct };
        }



        // Reformat incoming clinical driver cards to be extremely compact with clinical icons and level gauges
        function formatClinicalCards(htmlString, isAbnormal = false) {
            if (!htmlString) return "";
            const temp = document.createElement('div');
            temp.innerHTML = htmlString;
            const cards = temp.querySelectorAll('.card');
            let formattedHtml = "";

            cards.forEach(card => {
                const h4 = card.querySelector('h4');
                let titleText = h4 ? h4.textContent.trim() : "";

                // Strip numbers like "1. ", "2. " from the beginning
                titleText = titleText.replace(/^\d+\.\s*/, "");

                const pElements = card.querySelectorAll('p');
                let valueText = "";
                let interpretationText = "";

                if (pElements.length > 0) {
                    valueText = pElements[0].textContent.replace('Patient Value:', 'Val:').replace('Ideal Range:', 'Ideal:').trim();
                }
                if (pElements.length > 1) {
                    interpretationText = pElements[1].textContent.replace('Interpretation:', '').trim();
                } else if (card.querySelector('.interpretation')) {
                    interpretationText = card.querySelector('.interpretation').textContent.replace('Interpretation:', '').trim();
                }

                // Extract feature name and clean raw value to pass to getFeatureVisualProps
                const featureKey = titleText.split('|')[0].trim().toLowerCase();
                const readableTitle = titleText;

                let rawVal = 0;
                const matchVal = valueText.match(/Val:\s*([0-9.]+)/i);
                if (matchVal) {
                    rawVal = parseFloat(matchVal[1]);
                }

                const { iconSvg, statusClass, pct } = getFeatureVisualProps(featureKey, rawVal);
                const cardClass = isAbnormal ? "card abnormal-card compact-report-card" : "card driver-card compact-report-card";

                formattedHtml += `
                    <div class="${cardClass}">
                        <div class="d-flex align-items-center justify-content-between">
                            <div class="d-flex align-items-center gap-2">
                                <div class="clinical-icon-wrapper">
                                    ${iconSvg}
                                </div>
                                <span class="fw-bold tracking-wider text-uppercase" style="font-size: 0.8rem; letter-spacing: 0.5px; color: var(--text-primary);">${readableTitle}</span>
                            </div>
                            <span class="badge-vitals" style="font-size: 0.72rem; color: var(--text-secondary); background: rgba(255,255,255,0.03); padding: 2px 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); font-family: monospace;">${valueText}</span>
                        </div>
                        <div class="interpretation-text text-truncate-2-lines" style="font-size: 0.76rem; color: var(--text-secondary); font-style: italic; line-height: 1.45; margin-top: 4px;" title="${interpretationText}">
                            ${interpretationText}
                        </div>
                        <!-- Horizontal Status Gauge Bar -->
                        <div class="level-gauge-bg mt-2">
                            <div class="level-gauge-fill ${statusClass}" style="width: ${pct}%"></div>
                        </div>
                    </div>
                `;
            });
            return formattedHtml;
        }

        // Populate clinical flags cards with API results and animations
        function populateAbnormalFlags(clinical_details) {
            const primaryContainer = document.getElementById('display-primary-drivers');
            const secondaryContainer = document.getElementById('display-secondary-anomalies');
            const narrativeContainer = document.getElementById('display-display-narrative') || document.getElementById('display-narrative');
            const narrativeCard = document.getElementById('narrative-summary-card');

            if (primaryContainer) primaryContainer.innerHTML = "";
            if (secondaryContainer) secondaryContainer.innerHTML = "";
            if (narrativeContainer) narrativeContainer.innerHTML = "";

            if (!clinical_details) return;

            // 1. Populate Narrative Summary (with typewriter effect)
            if (clinical_details.narrative) {
                if (narrativeCard) narrativeCard.style.display = "block";
                if (narrativeContainer) {
                    typewriterEffect(narrativeContainer, clinical_details.narrative, 8);
                }
            } else {
                if (narrativeCard) narrativeCard.style.display = "none";
            }

            // 2. Populate Primary Drivers
            if (clinical_details.primary_html && primaryContainer) {
                const formattedPrimary = formatClinicalCards(clinical_details.primary_html, false);
                primaryContainer.innerHTML = formattedPrimary;
            }

            // 3. Populate Secondary Anomalies
            if (secondaryContainer) {
                if (clinical_details.secondary_html && clinical_details.secondary_html.indexOf('abnormal-card') !== -1) {
                    const formattedSecondary = formatClinicalCards(clinical_details.secondary_html, true);
                    secondaryContainer.innerHTML = formattedSecondary;
                } else {
                    secondaryContainer.innerHTML = `
                        <div class="text-secondary fs-7 py-4 px-2 border border-secondary-subtle border-dashed rounded text-center" style="border-style: dashed !important; border-width: 1px !important; opacity: 0.65; border-radius: 12px !important;">
                            No secondary physiological abnormalities detected outside primary drivers.
                        </div>
                    `;
                }
            }

            // 4. Staggered fade-in-up animations for cards
            const allCards = document.querySelectorAll('.report-grid-column .card');
            allCards.forEach((card, index) => {
                card.classList.add('fade-in-up');
                card.style.animationDelay = `${(index + 2) * 0.12}s`;
            });
        }

        // Trigger Plotly resizes when changing results dashboard tabs to resolve hidden-container layout calculations
        document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tabTrigger => {
            tabTrigger.addEventListener('shown.bs.tab', (e) => {
                const targetId = e.target.getAttribute('data-bs-target');
                if (targetId === '#panel-analytics-bar') {
                    Plotly.Plots.resize('plotly-bar-chart');
                } else if (targetId === '#panel-analytics-gaf') {
                    Plotly.Plots.resize('plotly-gradcam-heatmap');
                }
            });
        });

        // Initialize Bootstrap tooltips for clinical helper labels
        document.addEventListener('DOMContentLoaded', () => {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });

            // Play sound ticks on form control change/input
            document.querySelectorAll('#diagnosis-form input[type="range"]').forEach(input => {
                input.addEventListener('input', () => {
                    SciFiAudioController.playInputTick();
                });
            });

            document.querySelectorAll('#diagnosis-form select').forEach(select => {
                select.addEventListener('change', () => {
                    SciFiAudioController.playInputTick();
                });
            });

            // Play button clicks
            document.querySelectorAll('button, .btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    SciFiAudioController.playButtonClick();
                });
            });

            // Generate Data Stream for Scanner Overlay
            const streamContainer = document.getElementById('data-stream-container');
            if (streamContainer) {
                const cols = Math.floor(window.innerWidth / 30);
                for (let i = 0; i < cols; i++) {
                    const col = document.createElement('div');
                    col.className = 'data-stream-column';
                    col.style.left = `${(i / cols) * 100}%`;
                    col.style.animationDuration = `${2 + Math.random() * 3}s`;
                    col.style.animationDelay = `${Math.random() * 2}s`;
                    col.style.opacity = `${0.3 + Math.random() * 0.7}`;

                    // Generate random hex/binary content
                    let content = '';
                    const chars = '0123456789ABCDEF01';
                    const len = 10 + Math.floor(Math.random() * 20);
                    for (let j = 0; j < len; j++) {
                        content += chars.charAt(Math.floor(Math.random() * chars.length));
                        if (Math.random() > 0.8) content += ' '; // occasional gap
                    }
                    col.textContent = content;
                    streamContainer.appendChild(col);
                }
            }

            // --- PATIENT SEARCH MODAL LOGIC ---
            const searchInput = document.getElementById('patientSearchInput');
            const searchResults = document.getElementById('searchResults');
            const searchBtn = document.getElementById('btnSearchPatient');

            function performSearch() {
                const query = searchInput.value.trim();
                if (!query) return;

                searchResults.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-info" role="status"></div></div>';

                // Frontend Prototype: Mock empty search (no backend)
                Promise.resolve([])
                    .then(data => {
                        if (data.error) throw new Error(data.error);
                        if (data.length === 0) {
                            searchResults.innerHTML = '<div class="text-center py-4 text-secondary">No patients found.</div>';
                            return;
                        }

                        let html = '';
                        data.forEach(patient => {
                            html += `
                                <div class="card glass-card border-0 mb-3 p-3">
                                    <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom border-secondary">
                                        <div>
                                            <h5 class="m-0 text-info font-title">${patient.name}</h5>
                                            <small class="text-secondary">${patient.hospital_patient_id}</small>
                                        </div>
                                        <div class="text-end text-secondary fs-8">
                                            Visit Date: ${new Date(patient.created_at).toLocaleString()}
                                        </div>
                                    </div>
                            `;

                            if (patient.history && patient.history.length > 0) {
                                html += `<div class="timeline mt-2">`;
                                patient.history.forEach(visit => {
                                    const riskClass = visit.prediction_class === 1 ? 'text-danger' : 'text-success';
                                    const riskText = visit.prediction_class === 1 ? 'High Risk' : 'Low Risk';
                                    const riskBadge = visit.prediction_class === 1 ? 'bg-danger' : 'bg-success';
                                    html += `
                                        <div class="d-flex justify-content-between align-items-center mb-2 p-2 rounded" style="background: rgba(0,0,0,0.2);">
                                            <div>
                                                <div class="fw-bold fs-7">${new Date(visit.recorded_at).toLocaleString()}</div>
                                                <div class="fs-8 text-secondary">Age: ${visit.age}, Sex: ${visit.sex}</div>
                                            </div>
                                            <div class="text-end">
                                                <div class="badge ${riskBadge} bg-opacity-25 ${riskClass} border border-current mb-1">${riskText} (${(visit.risk_probability * 100).toFixed(1)}%)</div>
                                                ${visit.pdf_report_path ? `<div><a href="${visit.pdf_report_path}" target="_blank" class="btn btn-sm btn-outline-info py-0 px-2 fs-8">View Report</a></div>` : ''}
                                            </div>
                                        </div>
                                    `;
                                });
                                html += `</div>`;
                            } else {
                                html += `<div class="text-secondary fs-8">No clinical records found.</div>`;
                            }
                            html += `</div>`;
                        });
                        searchResults.innerHTML = html;
                    })
                    .catch(err => {
                        searchResults.innerHTML = `<div class="text-danger p-3 bg-danger bg-opacity-10 rounded border border-danger">Error: ${err.message}</div>`;
                    });
            }

            searchBtn?.addEventListener('click', performSearch);
            searchInput?.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') performSearch();
            });
            // ----------------------------------

        });
