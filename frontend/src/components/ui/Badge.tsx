import type { ReactNode } from "react";

export default function Badge({
  children,
  color = "#0a84ff",
  className = "",
}: {
  children: ReactNode;
  color?: string;
  className?: string;
}) {
  return (
    <span
      className={`inline-block rounded-md px-3 py-[3px] text-[13px] font-bold text-white ${className}`}
      style={{ background: color }}
    >
      {children}
    </span>
  );
}
