import { createRoot } from "react-dom/client";

import { ReactRuntimeProvider } from "./main.jsx";
import { TutorialPage } from "./tutorial/TutorialPage.jsx";
import "./styles/tutorial.css";

createRoot(document.getElementById("root")).render(<ReactRuntimeProvider><TutorialPage /></ReactRuntimeProvider>);
