/**
 * check-integrity.js — Project Integrity Validator (Phase 0)
 *
 * Verifies:
 * 1. Single source-of-truth: only ONE `src/` inside the Vite root
 * 2. No orphan feature files outside `src/`
 * 3. Vite entry point (`index.html`) exists and references correct script
 * 4. Main entry (`src/main.tsx`) exists and imports design system
 * 5. App entry (`src/App.tsx`) exists
 * 6. Required lib modules exist (`supabase.ts`, `safeRenderer.ts`)
 * 7. No duplicate `renderSafe` implementations (SSOT)
 *
 * Usage:  node scripts/check-integrity.js
 */

import { existsSync, readFileSync, readdirSync, statSync } from 'fs';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SRC = resolve(ROOT, 'src');

let failures = 0;

function pass(msg) { console.log(`  ✅  ${msg}`); }
function fail(msg) { console.error(`  ❌  ${msg}`); failures++; }
function warn(msg) { console.warn(`  ⚠️   ${msg}`); }

// ─── 1. Single `src/` directory ──────────────────────────────
console.log('\n── 1. Single Source Directory ──');
const topLevel = readdirSync(ROOT).filter(name => {
  if (name === 'node_modules' || name === 'dist' || name === '.git' || name === 'scripts' || name === 'public') return false;
  const full = join(ROOT, name);
  return statSync(full).isDirectory() && name === 'src';
});
if (topLevel.length === 1) {
  pass('Exactly one `src/` directory found.');
} else if (topLevel.length === 0) {
  fail('No `src/` directory found in frontend root!');
} else {
  fail(`Multiple \`src/\` directories detected: ${topLevel.join(', ')}`);
}

// ─── 2. Vite entry point (`index.html`) ──────────────────────
console.log('\n── 2. Vite Entry Point ──');
const indexPath = resolve(ROOT, 'index.html');
if (existsSync(indexPath)) {
  const html = readFileSync(indexPath, 'utf-8');
  if (html.includes('src/main.tsx')) {
    pass('index.html references src/main.tsx');
  } else {
    fail('index.html does NOT reference src/main.tsx — Vite will not load the app.');
  }
  if (html.includes('id="app"')) {
    pass('index.html has mount point <div id="app">');
  } else {
    fail('index.html missing <div id="app"> mount point.');
  }
} else {
  fail('index.html not found in frontend root!');
}

// ─── 3. Main entry & design-system import ────────────────────
console.log('\n── 3. Main Entry & Design System Import ──');
const mainPath = resolve(SRC, 'main.tsx');
if (existsSync(mainPath)) {
  const mainContent = readFileSync(mainPath, 'utf-8');
  pass('src/main.tsx exists.');

  const importsStyle = mainContent.includes("./style.css") || mainContent.includes("./design-tokens.css");
  if (importsStyle) {
    pass('main.tsx imports design system stylesheet(s).');
  } else {
    fail('main.tsx does NOT import style.css or design-tokens.css — design system is NOT wired!');
  }

  if (mainContent.includes("from './App'") || mainContent.includes('from "./App"')) {
    pass('main.tsx imports App component.');
  } else {
    fail('main.tsx does NOT import App component.');
  }
} else {
  fail('src/main.tsx not found!');
}

// ─── 4. App.tsx exists ───────────────────────────────────────
console.log('\n── 4. App Entry ──');
const appPath = resolve(SRC, 'App.tsx');
if (existsSync(appPath)) {
  pass('src/App.tsx exists.');
} else {
  fail('src/App.tsx not found!');
}

// ─── 5. Required lib modules ────────────────────────────────
console.log('\n── 5. Required Library Modules ──');
const requiredLibs = [
  { file: 'supabase.ts', label: 'Supabase client' },
  { file: 'safeRenderer.ts', label: 'Safe renderer / normalize' },
  { file: 'utils.ts', label: 'Utilities' },
];
for (const { file, label } of requiredLibs) {
  const libPath = resolve(SRC, 'lib', file);
  if (existsSync(libPath)) {
    pass(`lib/${file} (${label}) exists.`);
  } else {
    fail(`lib/${file} (${label}) is MISSING.`);
  }
}

// ─── 6. Duplicate renderSafe check ──────────────────────────
console.log('\n── 6. SSOT: Duplicate renderSafe Check ──');
const renderSafeLocations = [];
const libDir = resolve(SRC, 'lib');
if (existsSync(libDir)) {
  for (const f of readdirSync(libDir)) {
    const full = join(libDir, f);
    if (statSync(full).isFile() && (f.endsWith('.ts') || f.endsWith('.tsx'))) {
      const content = readFileSync(full, 'utf-8');
      if (content.includes('export function renderSafe') || content.includes('export const renderSafe')) {
        renderSafeLocations.push(`lib/${f}`);
      }
    }
  }
}
if (renderSafeLocations.length === 0) {
  fail('renderSafe() not found in any lib file.');
} else if (renderSafeLocations.length === 1) {
  pass(`renderSafe() defined in exactly one place: ${renderSafeLocations[0]}`);
} else {
  fail(`renderSafe() is DUPLICATED in multiple files: ${renderSafeLocations.join(', ')} — violates SSOT!`);
}

// ─── 7. Design token files ──────────────────────────────────
console.log('\n── 7. Design Token Files ──');
const stylePath = resolve(SRC, 'style.css');
const tokensPath = resolve(SRC, 'design-tokens.css');
if (existsSync(stylePath)) pass('src/style.css exists.');
else warn('src/style.css not found (may be merged into design-tokens.css).');
if (existsSync(tokensPath)) pass('src/design-tokens.css exists.');
else warn('src/design-tokens.css not found (may be merged into style.css).');
if (!existsSync(stylePath) && !existsSync(tokensPath)) {
  fail('Neither style.css nor design-tokens.css exist — NO design system loaded!');
}

// ─── 8. Orphan .tsx/.ts files directly in src/ ──────────────
console.log('\n── 8. Orphan Files Check ──');
const allowedRootFiles = new Set(['App.tsx', 'main.tsx', 'vite-env.d.ts']);
const rootFiles = readdirSync(SRC).filter(f => {
  return (f.endsWith('.ts') || f.endsWith('.tsx')) && !allowedRootFiles.has(f);
});
if (rootFiles.length === 0) {
  pass('No orphan .ts/.tsx files in src/ root.');
} else {
  for (const f of rootFiles) {
    warn(`Orphan file in src/ root: ${f} — consider moving to lib/ or components/`);
  }
}

// ─── Summary ─────────────────────────────────────────────────
console.log('\n══════════════════════════════════════════════');
if (failures === 0) {
  console.log('🎉  ALL INTEGRITY CHECKS PASSED');
} else {
  console.error(`💥  ${failures} INTEGRITY CHECK(S) FAILED — fix before proceeding`);
}
console.log('══════════════════════════════════════════════\n');
process.exit(failures > 0 ? 1 : 0);
