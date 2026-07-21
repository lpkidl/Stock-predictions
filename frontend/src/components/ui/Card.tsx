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
      className={`rounded-card border border-hairline bg-surface p-6 transition-[border-color,transform,box-shadow] duration-200 hover:border-white/[0.14] ${className}`}
    >
      {children}
    </div>
  );
}
