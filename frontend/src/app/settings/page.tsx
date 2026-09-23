import { ProviderKeysPanel } from "@/features/settings/ProviderKeysPanel";
import { ProviderTaskTable } from "@/features/settings/ProviderTaskTable";
import { RiskSettingsPanel } from "@/features/settings/RiskSettingsPanel";
import { MenuButton } from "@/shared/ui/NavigationDrawer";

export const metadata = { title: "Settings — AI Trading Bot" };

export default function SettingsPage() {
  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-3 border-b border-line px-4 py-2">
        <MenuButton />
        <div>
          <h1 className="text-base font-bold">AI provider settings</h1>
          <p className="text-xs text-ink-muted">
            Pick which provider and model runs each AI task — takes effect immediately, no backend
            restart required.
          </p>
        </div>
      </header>
      <main className="min-h-0 flex-1 overflow-y-auto">
        <section>
          <h2 className="px-4 pt-3 text-sm font-semibold text-ink">API keys</h2>
          <ProviderKeysPanel />
        </section>
        <section>
          <h2 className="px-4 pt-3 text-sm font-semibold text-ink">Task provider &amp; model</h2>
          <ProviderTaskTable />
        </section>
        <section>
          <h2 className="px-4 pt-3 text-sm font-semibold text-ink">Risk</h2>
          <RiskSettingsPanel />
        </section>
      </main>
    </div>
  );
}
