import type { ReactNode } from "react";

export default function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-card border border-hairline bg-surface p-6 transition-[border-color,transform] duration-150 ${className}`}
    >
      {children}
    </div>
  );
}
