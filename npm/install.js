#!/usr/bin/env node

const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const VERSION = require('./package.json').version;
const REPO = 'yurvon-screamo/smos';

function getTarget() {
  const platform = process.platform;
  const arch = process.arch;

  const map = {
    'win32-x64': 'x86_64-pc-windows-msvc',
    'linux-x64': 'x86_64-unknown-linux-gnu',
    'darwin-x64': 'x86_64-apple-darwin',
    'darwin-arm64': 'aarch64-apple-darwin',
  };

  const key = `${platform}-${arch}`;
  const target = map[key];
  if (!target) {
    console.error(`Unsupported platform: ${key}`);
    console.error('Please install from source: cargo install --git https://github.com/yurvon-screamo/smos');
    process.exit(1);
  }
  return target;
}

function getArchiveExt(target) {
  if (target.includes('windows')) return 'zip';
  return 'tar.gz';
}

async function downloadBinary() {
  const target = getTarget();
  const ext = getArchiveExt(target);
  const archiveName = `smos-${target}-v${VERSION}.${ext}`;
  const downloadUrl = `https://github.com/${REPO}/releases/download/v${VERSION}/${archiveName}`;

  const binDir = path.join(__dirname, 'bin');
  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }

  const archivePath = path.join(binDir, archiveName);

  console.log(`Downloading smos v${VERSION} for ${target}...`);
  console.log(`  URL: ${downloadUrl}`);

  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(archivePath);

    function handleResponse(res) {
      if (res.statusCode === 302 || res.statusCode === 301) {
        https.get(res.headers.location, handleResponse).on('error', reject);
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`));
        return;
      }
      res.pipe(file);
      file.on('finish', () => {
        file.close(() => {
          console.log(`Extracting ${archiveName}...`);
          const binName = target.includes('windows') ? 'smos.exe' : 'smos';
          const binPath = path.join(binDir, binName);

          if (ext === 'zip') {
            try {
              execSync(`tar -xf "${archivePath}" -C "${binDir}"`, { stdio: 'inherit' });
            } catch {
              execSync(`powershell Expand-Archive -Path "${archivePath}" -DestinationPath "${binDir}" -Force`, { stdio: 'inherit' });
            }
          } else {
            execSync(`tar -xzf "${archivePath}" -C "${binDir}"`, { stdio: 'inherit' });
          }

          if (!target.includes('windows')) {
            fs.chmodSync(binPath, 0o755);
          }

          fs.unlinkSync(archivePath);
          console.log(`smos v${VERSION} installed successfully!`);
          resolve();
        });
      });
    }

    https.get(downloadUrl, handleResponse).on('error', reject);
  });
}

downloadBinary().catch(err => {
  console.error('Installation failed:', err.message);
  console.error('');
  console.error('Alternative: install from source');
  console.error('  cargo install --git https://github.com/yurvon-screamo/smos');
  process.exit(1);
});
