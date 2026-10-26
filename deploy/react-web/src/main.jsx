import { MantineProvider } from "@mantine/core";

export function ReactRuntimeProvider({ children }) {
  return (
    <MantineProvider
      theme={{
        primaryColor: "teal",
        fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        fontFamilyMonospace: "ui-monospace, SFMono-Regular, Menlo, monospace"
      }}
      withCssVariables={false}
    >
      {children}
    </MantineProvider>
  );
}
