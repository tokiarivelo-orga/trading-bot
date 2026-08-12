import fs from "fs";
import path from "path";
import Link from "next/link";
import { MenuButton } from "@/shared/ui/NavigationDrawer";

// Reads the docs directory and the root directory for MD files
export default async function DocsIndexPage() {
  const rootDir = path.join(process.cwd(), "..");
  const docsDir = path.join(rootDir, "docs");

  const files: { slug: string; title: string; type: string }[] = [];

  // Read root READMEs
  try {
    const rootFiles = await fs.promises.readdir(rootDir);
    for (const file of rootFiles) {
      if (file.toLowerCase().endsWith(".md")) {
        files.push({
          slug: file.replace(".md", ""),
          title: file,
          type: "Root Documentation",
        });
      }
    }
  } catch (e) {
    console.error("Could not read root dir", e);
  }

  // Read docs/
  try {
    if (fs.existsSync(docsDir)) {
      const docFiles = await fs.promises.readdir(docsDir);
      for (const file of docFiles) {
        if (file.toLowerCase().endsWith(".md")) {
          files.push({
            slug: `docs_${file.replace(".md", "")}`, // We prefix with docs_ to differentiate
            title: file,
            type: "Technical Guides",
          });
        }
      }
    }
  } catch (e) {
    console.error("Could not read docs dir", e);
  }

  return (
    <div className="flex h-full flex-col bg-background text-ink min-h-screen">
      <header className="border-b border-line px-6 py-4 flex items-center gap-4 bg-panel/50 backdrop-blur-md sticky top-0 z-10">
        <MenuButton />
        <div>
          <h1 className="text-xl font-bold tracking-tight">
            Documentation Library
          </h1>
          <p className="text-xs text-ink-muted mt-1">Read guides and architectures</p>
        </div>
      </header>

      <main className="flex-1 p-8 max-w-4xl mx-auto w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {files.map((file) => (
            <Link 
              key={file.slug} 
              href={`/docs/${file.slug}`}
              className="bg-panel/40 border border-line rounded-xl p-6 hover:bg-panel hover:border-accent/50 hover:shadow-lg transition-all group"
            >
              <div className="flex items-center gap-3 mb-2">
                <svg className="w-6 h-6 text-accent group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <h2 className="text-lg font-bold text-ink group-hover:text-accent transition-colors">
                  {file.title}
                </h2>
              </div>
              <p className="text-sm text-ink-muted">{file.type}</p>
            </Link>
          ))}
          {files.length === 0 && (
            <div className="col-span-2 text-center text-ink-muted py-12">
              No markdown documents found in the project.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
