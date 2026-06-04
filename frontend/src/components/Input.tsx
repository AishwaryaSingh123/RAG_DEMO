import React from "react";

export default function Input({
  type = "text",
  value,
  onChange,
  placeholder,
  error,
}: {
  type?: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder?: string;
  error?: string;
}) {
  return (
    <div className="mb-4">
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className={`
          w-full rounded-md border border-[var(--border)] bg-[var(--color-bg-surface)]
          py-2 px-3 text-base text-[var(--color-text-primary)] focus:outline-none
          focus:border-[var(--color-accent)] focus:ring-2 focus:ring-[var(--color-accent)] transition
          ${error ? "border-[var(--color-danger)]" : ""}
        `}
      />
      {error && (
        <p className="mt-1 text-xs text-[var(--color-danger)]">{error}</p>
      )}
    </div>
  );
}
