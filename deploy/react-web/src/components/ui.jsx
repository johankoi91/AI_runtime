import { UnstyledButton } from "@mantine/core";

export function UiButton({ className = "", children, ...props }) {
  return (
    <UnstyledButton unstyled className={className || undefined} {...props}>
      {children}
    </UnstyledButton>
  );
}

export function Badge({ children, kind = "", ...props }) {
  return <span className={`badge ${kind}`} {...props}>{children}</span>;
}

export function Facts({ rows, ...props }) {
  return (
    <div {...props}>
      {rows.map(([label, value]) => (
        <div className="fact" key={`${label}-${value}`}>
          <span>{label}</span>
          <b>{value}</b>
        </div>
      ))}
    </div>
  );
}
