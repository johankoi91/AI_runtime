import { createRoot } from "react-dom/client";

import { ReactRuntimeProvider } from "./main.jsx";
import { TracePage } from "./inference/TracePage.jsx";
import "./styles/inference.css";

createRoot(document.getElementById("root")).render(<ReactRuntimeProvider><TracePage /></ReactRuntimeProvider>);
