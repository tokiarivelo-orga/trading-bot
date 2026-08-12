import fs from "fs";
import path from "path";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MenuButton } from "@/shared/ui/NavigationDrawer";

// Force dynamic or dynamicParams to true if needed, but Next.js will handle this.
export const dynamic = "force-dynamic";

export default async function DocViewPage({ params }: { params: Promise<{ slug: string }> }) {
  const rootDir = path.join(process.cwd(), "..");
  const docsDir = path.join(rootDir, "docs");

  const resolvedParams = await params;
  const slug = resolvedParams.slug;
  let filePath = "";
  let fileTitle = "";

  if (slug.startsWith("docs_")) {
    const realName = slug.replace("docs_", "") + ".md";
    filePath = path.join(docsDir, realName);
    fileTitle = `docs/${realName}`;
  } else {
    const realName = slug + ".md";
    filePath = path.join(rootDir, realName);
    fileTitle = realName;
  }

  let content = "";
  try {
    content = await fs.promises.readFile(filePath, "utf-8");
  } catch (e) {
    content = "# Error\nCould not load the requested document.";
  }

  return (
    <div className="flex h-full flex-col bg-background text-ink min-h-screen">
      <header className="border-b border-line px-6 py-4 flex items-center gap-4 bg-panel/50 backdrop-blur-md sticky top-0 z-10">
        <MenuButton />
        <Link 
          href="/docs"
          className="text-ink-muted hover:text-accent transition-colors flex items-center gap-1 text-sm font-medium"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
          Back to Docs
        </Link>
        <div className="ml-4 pl-4 border-l border-line">
          <h1 className="text-xl font-bold tracking-tight">
            {fileTitle}
          </h1>
        </div>
      </header>

      <main className="flex-1 p-8 w-full">
        <article className="max-w-4xl mx-auto bg-panel/40 border border-line rounded-xl p-8 shadow-xl">
          <div className="prose prose-invert prose-blue max-w-none 
            prose-headings:text-ink prose-headings:font-bold prose-headings:border-b prose-headings:border-line/50 prose-headings:pb-2
            prose-p:text-ink-muted prose-p:leading-relaxed
            prose-a:text-accent prose-a:no-underline hover:prose-a:underline
            prose-code:text-accent prose-code:bg-accent/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none
            prose-pre:bg-background prose-pre:border prose-pre:border-line prose-pre:p-4 prose-pre:rounded-lg
            prose-strong:text-ink prose-strong:font-bold
            prose-ul:text-ink-muted prose-ol:text-ink-muted
            prose-blockquote:border-l-4 prose-blockquote:border-accent prose-blockquote:bg-accent/5 prose-blockquote:py-1 prose-blockquote:px-4 prose-blockquote:not-italic prose-blockquote:text-ink-muted
            prose-table:w-full prose-table:text-left prose-table:border-collapse
            prose-th:border-b prose-th:border-line prose-th:py-2 prose-th:text-ink
            prose-td:border-b prose-td:border-line/50 prose-td:py-2"
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {content}
            </ReactMarkdown>
          </div>
        </article>
      </main>
    </div>
  );
}
