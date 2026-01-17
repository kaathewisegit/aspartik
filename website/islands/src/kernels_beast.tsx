import { render } from "solid-js/web"
import Kernel from "./components/Kernel"
import { getElementById } from "./utils"

render(() => <Kernel />, getElementById("kernelsBeast"))
