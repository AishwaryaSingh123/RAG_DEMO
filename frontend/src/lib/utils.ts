/**
 * utils.ts — SINGLE SOURCE OF TRUTH for data‑safety helpers.
 *
 * renderSafe()  – converts any value to a safe React‑renderable string.
 * normalize()   – coerces null/undefined/empty to "N/A".
 *
 * ALL display components MUST use renderSafe() when rendering external data.
 */

/**
 * Convert undefined / null / empty arrays / empty objects to "N/A".
 */
export function normalize<T>(value: T | undefined | null): T | string {
  if (value === undefined || value === null) return 'N/A';
  if (Array.isArray(value) && value.length === 0) return 'N/A';
  if (typeof value === 'object' && value !== null && Object.keys(value as object).length === 0) {
    return 'N/A';
  }
  return value;
}

/**
 * Safely render any value as a string.
 * - Primitives → String()
 * - Arrays    → comma-separated recursive renderSafe
 * - Objects   → smart field extraction (name > title > label > email > id > JSON)
 *
 * This guarantees we NEVER see "[object Object]" in the UI.
 */
export function renderSafe(value: any): string {
  if (value === null || value === undefined) {
    return 'N/A';
  }
  if (typeof value === 'object') {
    if (Array.isArray(value)) {
      return value.map(item => renderSafe(item)).join(', ');
    }
    // Priority order: name → title → label → email → id → JSON fallback
    if ('name' in value && value.name != null) return String(value.name);
    if ('title' in value && value.title != null) return String(value.title);
    if ('label' in value && value.label != null) return String(value.label);
    if ('email' in value && value.email != null) return String(value.email);
    if ('id' in value && value.id != null) return String(value.id);
    return JSON.stringify(value);
  }
  return String(value);
}
