import { createBrowserRouter } from "react-router-dom";
import { Root } from "./components/Root";
import { HomePage } from "./components/HomePage";
import { AboutPage } from "./components/AboutPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: HomePage },
      { path: "about", Component: AboutPage },
    ],
  },
]);
