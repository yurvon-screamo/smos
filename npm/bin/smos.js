#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const platform = process.platform;
const binName = platform === 'win32' ? 'smos.exe' : 'smos';
const binPath = path.join(__dirname, binName);

if (!fs.existsSync(binPath)) {
  console.error('smos binary not found. Run npm install to download it.');
  process.exit(1);
}

const child = spawn(binPath, process.argv.slice(2), {
  stdio: 'inherit',
  cwd: process.cwd()
});

child.on('exit', (code) => {
  process.exit(code);
});
