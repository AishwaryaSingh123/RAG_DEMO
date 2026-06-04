# UI Redesign and Layout Overhaul

## Goal Description
Redesign the entire UI layer of the RAG Demo application to a modern, consistent, production‑grade look while preserving all existing backend logic and Supabase authentication. The overhaul includes:
- Global design system (CSS variables, spacing, typography, colors)
- Consistent layout components (Header, Sidebar, Main content wrapper)
- Refreshed authentication pages (Login, Register, Forgot Password)
- Role‑based dashboards (Admin, User, Moderator) with responsive grids
- Accessible components with hover/focus micro‑animations
- Updated global stylesheet and component‑specific CSS modules

## User Review Required
> [!IMPORTANT]
> The plan introduces a new CSS‑variables based design system and restructures routing/layout. Please confirm you are comfortable with switching to a CSS‑variables design system and that you approve the creation of new layout components.

## Open Questions
> [!WARNING]
> 1. Do you prefer a dark‑mode default theme or a light theme with optional toggle?
> 2. Should we include a brand logo (provide image) or use a placeholder generated via the image tool?
> 3. Are there any specific color palettes or brand guidelines you want to follow?

## Proposed Changes
---
### Design System
- **[NEW]** `src/style.css` – define CSS variables for primary, secondary colors, spacing scale (8px base), typography (Google Font *Inter*), and utility classes for padding/margin.
- **[NEW]** `src/components/Container.tsx` – reusable max‑width container with centered layout.
- **[NEW]** `src/components/Header.tsx` – top navigation bar with brand name, user avatar, and logout.
- **[NEW]** `src/components/Sidebar.tsx` – collapsible side menu showing role‑specific links.
- **[NEW]** `src/layout/AppLayout.tsx` – combines Header, Sidebar, and content area; wraps protected routes.

### Authentication Pages
- **[MODIFY]** `src/pages/Login.tsx` – replace default HTML with styled form, add glass‑morphism card, focus states, and animated submit button.
- **[MODIFY]** `src/pages/Register.tsx` – similar styling, include password strength indicator.
- **[NEW]** `src/pages/ForgotPassword.tsx` – add simple email entry form with consistent styling.

### Role‑Based Dashboards
- **[NEW]** `src/pages/AdminDashboard.tsx`
- **[NEW]** `src/pages/UserDashboard.tsx`
- **[NEW]** `src/pages/ModeratorDashboard.tsx`
  Each dashboard uses a responsive CSS grid (min‑max 250px columns) with cards for stats, recent activity, and quick actions. Cards have subtle hover lift and shadow.

### Routing Adjustments
- **[MODIFY]** `src/App.tsx` – replace current route block with `<AppLayout>` wrapper for all protected routes. Keep the debug placeholder until styling is verified.
- Add a route redirect based on `session?.user?.role` (assumed role claim) to the appropriate dashboard.

### Accessibility & Micro‑animations
- Add `focus-visible` outlines, transition utilities in CSS, and `@keyframes` for button hover glow.
- Use `prefers-reduced-motion` media query to respect user settings.

## Source‑of‑Truth Rule
All UI must be built **ONLY** using files under:
- `/src/components/`
- `/src/lib/`
- `/src/pages/`

Do **NOT** use inline styling (`style={{}}`).
Do **NOT** duplicate utility implementations.
Do **NOT** keep old fallback CSS files; `style.css` must be replaced or explicitly merged.
Every feature must be imported in `App.tsx` or the route tree.

## Module Existence Validation
**Pre‑Execution Validation** (run before any implementation begins):
- Verify `src/style.css` (or merged design‑tokens) exists.
- Verify `src/lib/catalog.ts` exists and is imported where needed.
- Verify `src/lib/upload.ts` exists and is wired to UI components.
- Ensure `renderSafe()` is used in all display components.
- No orphan files outside `/src` directory.

## Import Wiring Requirement
- `main.tsx` must import the design system (e.g., `import "./style.css"`).
- `App.tsx` must use the updated `AppLayout` component.
- Catalog and upload modules must be imported in their respective UI entry points.
- Any file not imported is treated as **NON‑EXISTENT** and must be removed.

## Dev Sync Rule
After every major change:
1. Stop the Vite dev server.
2. Clear Vite cache (`rm -rf node_modules/.vite` or delete the `.vite` folder).
3. Restart the dev server (`npm run dev`).
4. Hard‑refresh the browser (Ctrl+Shift+R).

## Runtime Validation Checklist
- Confirm design tokens are visible in DevTools → computed styles.
- Confirm catalog API returns a non‑empty response.
- Confirm upload request hits `/api/ingest/file`.
- Ensure no `[object Object]` appears anywhere in the UI.
- Verify role‑based dashboards actually switch UI based on user role.

## Project Integrity Rule
- There must be **ONE** frontend source directory.
- Vite server root must match the edited directory (`/frontend`).
- Any duplicate `frontend/` or `src/` folders are INVALID.
- All features must be implemented in the active Vite root only.

## Verification Plan
### Automated Tests
- Run `npm run dev` and confirm the dev server starts without compilation warnings.
- Open `http://localhost:3001/` and verify the debug placeholder is visible.
- Manually navigate to each page to ensure layout renders correctly.

### Manual Verification
- Check authentication flow works (login → dashboard) without breaking Supabase calls.
- Confirm responsive behavior by resizing the browser.
- Review visual design against the proposed palette and spacing.

---
