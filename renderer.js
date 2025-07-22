const { ipcRenderer } = require('electron');
const path = require('path');

class SynthAnalyzerApp {
    constructor() {
        this.currentAudioFile = null;
        this.currentResults = null;
        this.isAnalyzing = false;

        this.initializeElements();
        this.setupEventListeners();
        this.checkSystemStatus();
    }

    initializeElements() {
        // File handling elements
        this.dropZone = document.getElementById('dropZone');
        this.selectFileBtn = document.getElementById('selectFileBtn');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.audioPreview = document.getElementById('audioPreview');
        
        // Analysis elements
        this.analysisControls = document.getElementById('analysisControls');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.progressSection = document.getElementById('progressSection');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.progressOutput = document.getElementById('progressOutput');
        
        // Results elements
        this.statusSection = document.getElementById('statusSection');
        this.resultsContent = document.getElementById('resultsContent');
        this.synthAudio = document.getElementById('synthAudio');
        this.synthAudioPlayer = document.getElementById('synthAudioPlayer');
        
        // Parameter elements
        this.paramElements = {
            oscType: document.getElementById('oscType'),
            filterType: document.getElementById('filterType'),
            filterCutoff: document.getElementById('filterCutoff'),
            filterResonance: document.getElementById('filterResonance'),
            attack: document.getElementById('attack'),
            decay: document.getElementById('decay'),
            sustain: document.getElementById('sustain'),
            release: document.getElementById('release'),
            amplitude: document.getElementById('amplitude')
        };
        
        // Action buttons
        this.copyParamsBtn = document.getElementById('copyParamsBtn');
        this.playSynthBtn = document.getElementById('playSynthBtn');
        this.openOutputBtn = document.getElementById('openOutputBtn');
        
        // Header buttons
        this.setupBtn = document.getElementById('setupBtn');
        this.trainBtn = document.getElementById('trainBtn');
        
        // Modals
        this.setupModal = document.getElementById('setupModal');
        this.trainingModal = document.getElementById('trainingModal');
        
        // Status
        this.statusText = document.getElementById('statusText');
        this.systemStatus = document.getElementById('systemStatus');
    }

    setupEventListeners() {
        // File handling
        this.dropZone.addEventListener('click', () => this.selectAudioFile());
        this.selectFileBtn.addEventListener('click', () => this.selectAudioFile());
        this.dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        this.dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.dropZone.addEventListener('drop', this.handleDrop.bind(this));
        
        // Analysis
        this.analyzeBtn.addEventListener('click', () => this.analyzeAudio());
        
        // Actions
        this.copyParamsBtn.addEventListener('click', () => this.copyParameters());
        this.playSynthBtn.addEventListener('click', () => this.playSynthesized());
        this.openOutputBtn.addEventListener('click', () => this.openOutputFolder());
        
        // Header buttons
        this.setupBtn.addEventListener('click', () => this.showSetupModal());
        this.trainBtn.addEventListener('click', () => this.showTrainingModal());
        
        // Modal handling
        this.setupModalHandlers();
        this.trainingModalHandlers();
        
        // IPC listeners
        this.setupIpcListeners();
    }

    setupModalHandlers() {
        const closeSetupModal = document.getElementById('closeSetupModal');
        const closeSetupBtn = document.getElementById('closeSetupBtn');
        const recheckSetupBtn = document.getElementById('recheckSetupBtn');
        
        closeSetupModal.addEventListener('click', () => this.hideSetupModal());
        closeSetupBtn.addEventListener('click', () => this.hideSetupModal());
        recheckSetupBtn.addEventListener('click', () => this.checkSystemStatus());
        
        // Close modal when clicking outside
        this.setupModal.addEventListener('click', (e) => {
            if (e.target === this.setupModal) this.hideSetupModal();
        });
    }

    trainingModalHandlers() {
        const closeTrainingModal = document.getElementById('closeTrainingModal');
        const cancelTrainingBtn = document.getElementById('cancelTrainingBtn');
        const startTrainingBtn = document.getElementById('startTrainingBtn');
        
        closeTrainingModal.addEventListener('click', () => this.hideTrainingModal());
        cancelTrainingBtn.addEventListener('click', () => this.hideTrainingModal());
        startTrainingBtn.addEventListener('click', () => this.startTraining());
        
        // Close modal when clicking outside
        this.trainingModal.addEventListener('click', (e) => {
            if (e.target === this.trainingModal) this.hideTrainingModal();
        });
    }

    setupIpcListeners() {
        ipcRenderer.on('analysis-progress', (event, data) => {
            this.updateAnalysisProgress(data);
        });

        ipcRenderer.on('training-progress', (event, data) => {
            this.updateTrainingProgress(data);
        });

        ipcRenderer.on('training-output', (event, data) => {
            this.appendTrainingOutput(data);
        });
    }

    // File Handling
    async selectAudioFile() {
        try {
            const filePath = await ipcRenderer.invoke('select-audio-file');
            if (filePath) {
                this.loadAudioFile(filePath);
            }
        } catch (error) {
            this.showError('Failed to select file', error.message);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        this.dropZone.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.loadAudioFile(files[0].path);
        }
    }

    loadAudioFile(filePath) {
        this.currentAudioFile = filePath;
        
        // Update UI
        this.fileName.textContent = path.basename(filePath);
        
        // Fix audio preview path for Electron
        this.audioPreview.src = this.fixAudioPath(filePath);
        
        // Show file info and analysis controls
        this.fileInfo.style.display = 'block';
        this.analysisControls.style.display = 'block';
        
        // Update status
        this.updateStatus('Audio file loaded. Ready to analyze.');
        
        // Hide previous results
        this.resultsContent.style.display = 'none';
        this.synthAudio.style.display = 'none';
    }

    // Analysis
    async analyzeAudio() {
        if (!this.currentAudioFile) {
            this.showError('No audio file selected');
            return;
        }

        if (this.isAnalyzing) {
            return;
        }

        this.isAnalyzing = true;
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.textContent = '🔄 Analyzing...';
        
        // Show progress
        this.progressSection.style.display = 'block';
        this.progressFill.style.width = '10%';
        this.progressText.textContent = 'Starting analysis...';
        this.progressOutput.textContent = '';

        try {
            const result = await ipcRenderer.invoke('analyze-audio', this.currentAudioFile);
            
            if (result.success) {
                this.displayResults(result);
                this.updateStatus('Analysis completed successfully');
            } else {
                this.showError('Analysis failed', result.error);
                if (result.stdout) {
                    console.log('Python stdout:', result.stdout);
                }
                if (result.stderr) {
                    console.error('Python stderr:', result.stderr);
                }
            }
        } catch (error) {
            this.showError('Analysis error', error.message);
        } finally {
            this.isAnalyzing = false;
            this.analyzeBtn.disabled = false;
            this.analyzeBtn.textContent = '🔍 Analyze Sample';
            this.progressSection.style.display = 'none';
        }
    }

    updateAnalysisProgress(data) {
        this.progressOutput.textContent += data;
        this.progressOutput.scrollTop = this.progressOutput.scrollHeight;
        
        // Update progress bar based on keywords in output
        let progress = 10;
        if (data.includes('Extracting')) progress = 30;
        if (data.includes('features')) progress = 50;
        if (data.includes('Predicting')) progress = 70;
        if (data.includes('Generating')) progress = 90;
        
        this.progressFill.style.width = `${progress}%`;
    }

    displayResults(result) {
        this.currentResults = result;
        
        // Update parameter values
        const params = result.results;
        
        this.paramElements.oscType.textContent = params.osc_type || '-';
        this.paramElements.filterType.textContent = params.filter_type || '-';
        this.paramElements.filterCutoff.textContent = params.filter_cutoff ? 
            `${params.filter_cutoff.toFixed(1)} Hz` : '-';
        this.paramElements.filterResonance.textContent = params.filter_resonance ? 
            params.filter_resonance.toFixed(2) : '-';
        this.paramElements.attack.textContent = params.attack ? 
            `${params.attack.toFixed(3)} s` : '-';
        this.paramElements.decay.textContent = params.decay ? 
            `${params.decay.toFixed(3)} s` : '-';
        this.paramElements.sustain.textContent = params.sustain ? 
            params.sustain.toFixed(2) : '-';
        this.paramElements.release.textContent = params.release ? 
            `${params.release.toFixed(3)} s` : '-';
        this.paramElements.amplitude.textContent = params.amplitude ? 
            params.amplitude.toFixed(2) : '-';
        
        // Show results
        this.resultsContent.style.display = 'block';
        
        // Handle synthesized audio if available
        if (result.synthAudioPath) {
            this.synthAudioPlayer.src = this.fixAudioPath(result.synthAudioPath);
            this.synthAudio.style.display = 'block';
            this.playSynthBtn.style.display = 'inline-block';
        }
        
        // Show output folder button
        if (result.outputDir) {
            this.openOutputBtn.style.display = 'inline-block';
            this.openOutputBtn.onclick = () => {
                ipcRenderer.invoke('open-file-location', result.outputDir);
            };
        }
    }

    // Actions
    copyParameters() {
        if (!this.currentResults) {
            this.showError('No parameters to copy');
            return;
        }

        const params = JSON.stringify(this.currentResults.results, null, 2);
        navigator.clipboard.writeText(params).then(() => {
            this.copyParamsBtn.textContent = '✅ Copied!';
            setTimeout(() => {
                this.copyParamsBtn.textContent = '📋 Copy JSON';
            }, 2000);
        }).catch(err => {
            this.showError('Failed to copy to clipboard', err.message);
        });
    }

    playSynthesized() {
        if (this.synthAudioPlayer.src) {
            this.synthAudioPlayer.currentTime = 0; // Reset to beginning
            this.synthAudioPlayer.play().catch(err => {
                console.error('Error playing synthesized audio:', err);
                this.showError('Playback failed', 'Could not play synthesized audio');
            });
        }
    }

    // Utility method to fix audio file paths for Electron
    fixAudioPath(filePath) {
        // Convert Windows backslashes to forward slashes
        const normalizedPath = filePath.replace(/\\/g, '/');
        
        // Handle different path formats
        if (normalizedPath.startsWith('file://')) {
            return normalizedPath;
        } else if (normalizedPath.match(/^[A-Za-z]:/)) {
            // Windows absolute path
            return `file:///${normalizedPath}`;
        } else if (normalizedPath.startsWith('/')) {
            // Unix absolute path
            return `file://${normalizedPath}`;
        } else {
            // Relative path
            return `file://${path.resolve(normalizedPath)}`;
        }
    }

    openOutputFolder() {
        if (this.currentResults && this.currentResults.outputDir) {
            ipcRenderer.invoke('open-file-location', this.currentResults.outputDir);
        }
    }

    // System Status and Setup
    async checkSystemStatus() {
        this.updateStatus('Checking system setup...');
        this.systemStatus.textContent = 'Checking...';

        try {
            const result = await ipcRenderer.invoke('check-python-setup');
            
            if (result.success) {
                this.systemStatus.textContent = 'System OK';
                this.updateStatus('System ready for analysis');
                
                if (this.setupModal.style.display === 'flex') {
                    document.getElementById('setupResults').innerHTML = `
                        <div class="success">
                            <h3>✅ System Check Passed</h3>
                            <p>All required files found and Python is working correctly.</p>
                            <p>You can now analyze audio files!</p>
                        </div>
                    `;
                }
            } else {
                this.systemStatus.textContent = 'Setup Required';
                this.updateStatus('System setup required - click "Check Setup" for details');
                
                if (this.setupModal.style.display === 'flex') {
                    let errorHtml = `<div class="error"><h3>❌ System Check Failed</h3>`;
                    
                    if (result.needsTraining) {
                        errorHtml += `
                            <p><strong>Missing trained models.</strong></p>
                            <p>You need to train the ML models first. Click "Train Models" in the main interface.</p>
                        `;
                    } else if (result.missingFiles) {
                        errorHtml += `
                            <p><strong>Missing files:</strong></p>
                            <ul>${result.missingFiles.map(f => `<li>${f}</li>`).join('')}</ul>
                            <p>Make sure all Python scripts are in the same directory as this app.</p>
                        `;
                    } else {
                        errorHtml += `<p>${result.error}</p>`;
                    }
                    
                    errorHtml += `</div>`;
                    document.getElementById('setupResults').innerHTML = errorHtml;
                }
            }
        } catch (error) {
            this.systemStatus.textContent = 'Error';
            this.updateStatus('System check failed');
            this.showError('System check failed', error.message);
        }
    }

    // Modals
    showSetupModal() {
        this.setupModal.style.display = 'flex';
        this.checkSystemStatus();
    }

    hideSetupModal() {
        this.setupModal.style.display = 'none';
    }

    showTrainingModal() {
        this.trainingModal.style.display = 'flex';
        // Reset training progress
        document.getElementById('trainingProgress').style.display = 'none';
        document.getElementById('trainingProgressFill').style.width = '0%';
        document.getElementById('trainingOutput').textContent = '';
    }

    hideTrainingModal() {
        this.trainingModal.style.display = 'none';
    }

    async startTraining() {
        const progressDiv = document.getElementById('trainingProgress');
        const progressFill = document.getElementById('trainingProgressFill');
        const progressText = document.getElementById('trainingProgressText');
        const outputDiv = document.getElementById('trainingOutput');
        const startBtn = document.getElementById('startTrainingBtn');
        
        // Show progress and disable button
        progressDiv.style.display = 'block';
        startBtn.disabled = true;
        startBtn.textContent = 'Training...';
        
        try {
            const result = await ipcRenderer.invoke('run-training-pipeline');
            
            if (result.success) {
                progressFill.style.width = '100%';
                progressText.textContent = 'Training completed successfully!';
                outputDiv.textContent += '\n✅ Training pipeline completed successfully!';
                
                setTimeout(() => {
                    this.hideTrainingModal();
                    this.checkSystemStatus(); // Recheck system status
                }, 2000);
            } else {
                progressText.textContent = 'Training failed';
                outputDiv.textContent += `\n❌ Training failed: ${result.error}`;
            }
        } catch (error) {
            progressText.textContent = 'Training error';
            outputDiv.textContent += `\n❌ Training error: ${error.message}`;
        } finally {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Training';
        }
    }

    updateTrainingProgress(data) {
        const progressFill = document.getElementById('trainingProgressFill');
        const progressText = document.getElementById('trainingProgressText');
        
        const progress = (data.step / data.total) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Step ${data.step}/${data.total}: ${data.message}`;
    }

    appendTrainingOutput(data) {
        const outputDiv = document.getElementById('trainingOutput');
        outputDiv.textContent += data;
        outputDiv.scrollTop = outputDiv.scrollHeight;
    }

    // Utility methods
    updateStatus(message) {
        this.statusText.textContent = message;
    }

    showError(title, message = '') {
        console.error(title, message);
        this.updateStatus(`Error: ${title}`);
        
        // Show error in status section
        const statusItem = this.statusSection.querySelector('.status-item');
        statusItem.className = 'status-item error';
        statusItem.innerHTML = `
            <strong>${title}</strong>
            ${message ? `<br><small>${message}</small>` : ''}
        `;
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SynthAnalyzerApp();
});