const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess = null;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true,
            webSecurity: false  // Allow loading local files
        },
        icon: path.join(__dirname, 'assets', 'icon.png'),
        titleBarStyle: 'default',
        show: false
    });

    mainWindow.loadFile('index.html');

    // Show window when ready to prevent visual flash
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    // Open DevTools in development
    if (process.argv.includes('--dev')) {
        mainWindow.webContents.openDevTools();
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
        // Kill any running Python processes
        if (pythonProcess) {
            pythonProcess.kill();
        }
    });
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// IPC Handlers
ipcMain.handle('select-audio-file', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openFile'],
        filters: [
            { name: 'Audio Files', extensions: ['wav', 'mp3', 'flac', 'ogg', 'm4a'] },
            { name: 'All Files', extensions: ['*'] }
        ]
    });

    if (result.canceled) {
        return null;
    }

    return result.filePaths[0];
});

ipcMain.handle('check-python-setup', async () => {
    return new Promise((resolve) => {
        // Check if Python is available and required files exist
        const pythonFiles = [
            'inference.py',
            'audio_feature_extractor.py',
            'model.py',
            'basic_synth.py'
        ];

        const missingFiles = pythonFiles.filter(file => !fs.existsSync(file));
        
        if (missingFiles.length > 0) {
            resolve({
                success: false,
                error: `Missing files: ${missingFiles.join(', ')}`,
                missingFiles
            });
            return;
        }

        // Check if trained models exist
        const modelsDir = 'trained_models';
        if (!fs.existsSync(modelsDir)) {
            resolve({
                success: false,
                error: 'No trained models found. Please train models first.',
                needsTraining: true
            });
            return;
        }

        // Test Python execution
        const testProcess = spawn('python', ['--version']);
        
        testProcess.on('close', (code) => {
            if (code === 0) {
                resolve({ success: true });
            } else {
                resolve({
                    success: false,
                    error: 'Python not found or not working properly'
                });
            }
        });

        testProcess.on('error', (err) => {
            resolve({
                success: false,
                error: `Python execution error: ${err.message}`
            });
        });
    });
});

ipcMain.handle('analyze-audio', async (event, audioFilePath) => {
    return new Promise((resolve) => {
        if (pythonProcess) {
            pythonProcess.kill();
        }

        // Create output directory
        const outputDir = 'electron_output';
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir);
        }

        // Run Python inference script
        pythonProcess = spawn('python', [
            'inference.py',
            '--audio', audioFilePath,
            '--output_dir', outputDir
        ]);

        let stdout = '';
        let stderr = '';

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
            // Log to console for debugging
            console.log(data.toString());
            // Send progress updates to renderer
            mainWindow.webContents.send('analysis-progress', data.toString());
        });

        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        pythonProcess.on('close', (code) => {
            pythonProcess = null;

            if (code === 0) {
                // Try to read the results
                try {
                    const audioFileName = path.basename(audioFilePath, path.extname(audioFilePath));
                    const resultsFile = path.join(outputDir, `analysis_${audioFileName}.json`);
                    
                    if (fs.existsSync(resultsFile)) {
                        const results = JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
                        
                        // Look for synthesized audio file
                        const synthFile = path.join(outputDir, `synthesized_${audioFileName}.wav`);
                        const synthExists = fs.existsSync(synthFile);
                        
                        resolve({
                            success: true,
                            results: results.predicted_parameters,
                            features: results.extracted_features,
                            synthAudioPath: synthExists ? path.resolve(synthFile) : null,
                            outputDir: path.resolve(outputDir)
                        });
                    } else {
                        resolve({
                            success: false,
                            error: 'Analysis completed but no results file found',
                            stdout,
                            stderr
                        });
                    }
                } catch (err) {
                    resolve({
                        success: false,
                        error: `Error reading results: ${err.message}`,
                        stdout,
                        stderr
                    });
                }
            } else {
                resolve({
                    success: false,
                    error: `Python process failed with code ${code}`,
                    stdout,
                    stderr
                });
            }
        });

        pythonProcess.on('error', (err) => {
            pythonProcess = null;
            resolve({
                success: false,
                error: `Failed to start Python process: ${err.message}`
            });
        });
    });
});

ipcMain.handle('run-training-pipeline', async () => {
    return new Promise((resolve) => {
        let currentStep = 0;
        const steps = [
            { script: 'training_data_generation.py', name: 'Generating training data' },
            { script: 'audio_feature_extractor.py', name: 'Extracting features' },
            { script: 'model.py', name: 'Training models' }
        ];

        function runNextStep() {
            if (currentStep >= steps.length) {
                resolve({ success: true, message: 'Training pipeline completed successfully!' });
                return;
            }

            const step = steps[currentStep];
            mainWindow.webContents.send('training-progress', {
                step: currentStep + 1,
                total: steps.length,
                message: step.name
            });

            const process = spawn('python', [step.script]);

            process.stdout.on('data', (data) => {
                mainWindow.webContents.send('training-output', data.toString());
            });

            process.stderr.on('data', (data) => {
                mainWindow.webContents.send('training-output', data.toString());
            });

            process.on('close', (code) => {
                if (code === 0) {
                    currentStep++;
                    runNextStep();
                } else {
                    resolve({
                        success: false,
                        error: `Step "${step.name}" failed with code ${code}`
                    });
                }
            });

            process.on('error', (err) => {
                resolve({
                    success: false,
                    error: `Failed to run "${step.name}": ${err.message}`
                });
            });
        }

        runNextStep();
    });
});

ipcMain.handle('open-file-location', async (event, filePath) => {
    const { shell } = require('electron');
    shell.showItemInFolder(filePath);
});

ipcMain.handle('play-audio-file', async (event, filePath) => {
    // This will be handled by the renderer process using HTML5 audio
    return fs.existsSync(filePath);
});