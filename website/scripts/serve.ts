import { Server } from "@kaathewise/ssg"

const server = new Server({ pagesDir: "src/pages/", sourceDir: "src/", assetDir: "public/" })
await server.listen()
