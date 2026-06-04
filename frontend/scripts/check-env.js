/**
 * check-env.js — Environment Validation (Phase 0)
 *
 * Verifies that all required VITE_* environment variables are present
 * and non-empty before the dev server starts.
 *
 * Usage:  node scripts/check-env.js
 * Exit 0 = all good, Exit 1 = missing vars.
 */

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

// ─── Required variables ──────────────────────────────────────
const REQUIRED = [
  'VITE_SUPABASE_URL',
  'VITE_SUPABASE_ANON_KEY',
];

// ─── Parse .env file ─────────────────────────────────────────
function parseEnvFile(filePath) {
  if (!existsSync(filePath)) return {};
  const lines = readFileSync(filePath, 'utf-8').split('\n');
  const vars = {};
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    const val = trimmed.slice(eqIdx + 1).trim();
    vars[key] = val;
  }
  return vars;
}

// ─── Run ─────────────────────────────────────────────────────
const envPath = resolve(ROOT, '.env');
const envVars = { ...parseEnvFile(envPath), ...process.env };

const missing = [];
const empty = [];

for (const key of REQUIRED) {
  if (!(key in envVars)) {
    missing.push(key);
  } else if (!envVars[key] || envVars[key].trim() === '' || envVars[key] === 'MISSING_ANON_KEY') {
    empty.push(key);
  }
}

if (missing.length || empty.length) {
  console.error('\n╔══════════════════════════════════════════════╗');
  console.error('║   ❌  ENVIRONMENT VALIDATION FAILED           ║');
  console.error('╚══════════════════════════════════════════════╝\n');

  if (missing.length) {
    console.error(`  Missing variables (not defined at all):`);
    missing.forEach(k => console.error(`    • ${k}`));
  }
  if (empty.length) {
    console.error(`  Empty / placeholder variables:`);
    empty.forEach(k => console.error(`    • ${k} = "${envVars[k]}"`));
  }

  console.error(`\n  Fix: Add the missing values to ${envPath}\n`);
  process.exit(1);
}

console.log('✅  Environment validation passed — all required variables present.');
process.exit(0);
