import React from "react";

type Variant = "primary" | "secondary" | "danger";

export default function Button({
  children,
  onClick,
  variant = "primary",
  loading = false,
  disabled = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  variant?: Variant;
  loading?: boolean;
  disabled?: boolean;
}) {
  const colors = {
    primary:
      "bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white",
    secondary:
      "bg-[var(--color-bg-surface)] hover:bg-[var(--color-bg-hover)] text-[var(--color-text-primary)]",
    danger:
      "bg-[var(--color-danger)] hover:bg-[var(--color-danger)] text-white",
  };
  return (
    <button
      onClick={onClick}
      disabled={loading || disabled}
      className={`
        rounded-md px-4 py-2 text-sm font-medium transition
        ${colors[variant]} ${loading ? "opacity-60 cursor-not-allowed" : ""}
      `}
    >
      {loading ? (
        <svg
          className="animate-spin h-4 w-4 mr-2 inline-block"
          viewBox="0 0 24 24"
        >
          <circle
            className="stroke-current opacity-25"
            cx="12"
            cy="12"
            r="10"
            strokeWidth="4"
          />
          <path
            className="stroke-current opacity-75"
            fill="none"
            d="M4 12a8 8 0 018-8"
            strokeWidth="4"
          />
        </svg>
      ) : null}
      {children}
    </button>
  );
}
