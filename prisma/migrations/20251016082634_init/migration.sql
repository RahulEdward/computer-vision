-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_audit_logs" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "workspaceId" TEXT,
    "userId" TEXT,
    "action" TEXT NOT NULL,
    "resource" TEXT NOT NULL,
    "details" TEXT,
    "ipAddress" TEXT,
    "userAgent" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "new_audit_logs" ("action", "createdAt", "details", "id", "ipAddress", "resource", "userAgent", "userId", "workspaceId") SELECT "action", "createdAt", "details", "id", "ipAddress", "resource", "userAgent", "userId", "workspaceId" FROM "audit_logs";
DROP TABLE "audit_logs";
ALTER TABLE "new_audit_logs" RENAME TO "audit_logs";
CREATE TABLE "new_executions" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "workflowId" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "startedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" DATETIME,
    "duration" INTEGER,
    "input" TEXT,
    "output" TEXT,
    "error" TEXT,
    CONSTRAINT "executions_workflowId_fkey" FOREIGN KEY ("workflowId") REFERENCES "workflows" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_executions" ("completedAt", "duration", "error", "id", "input", "output", "startedAt", "status", "workflowId") SELECT "completedAt", "duration", "error", "id", "input", "output", "startedAt", "status", "workflowId" FROM "executions";
DROP TABLE "executions";
ALTER TABLE "new_executions" RENAME TO "executions";
CREATE TABLE "new_workflows" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "workspaceId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "nodes" TEXT NOT NULL,
    "edges" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'draft',
    "createdBy" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "workflows_workspaceId_fkey" FOREIGN KEY ("workspaceId") REFERENCES "workspaces" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_workflows" ("createdAt", "createdBy", "description", "edges", "id", "name", "nodes", "status", "updatedAt", "workspaceId") SELECT "createdAt", "createdBy", "description", "edges", "id", "name", "nodes", "status", "updatedAt", "workspaceId" FROM "workflows";
DROP TABLE "workflows";
ALTER TABLE "new_workflows" RENAME TO "workflows";
CREATE TABLE "new_workspaces" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "ownerId" TEXT NOT NULL,
    "plan" TEXT NOT NULL DEFAULT 'free',
    "customDomain" TEXT,
    "settings" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "workspaces_ownerId_fkey" FOREIGN KEY ("ownerId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_workspaces" ("createdAt", "customDomain", "id", "name", "ownerId", "plan", "settings", "slug", "updatedAt") SELECT "createdAt", "customDomain", "id", "name", "ownerId", "plan", "settings", "slug", "updatedAt" FROM "workspaces";
DROP TABLE "workspaces";
ALTER TABLE "new_workspaces" RENAME TO "workspaces";
CREATE UNIQUE INDEX "workspaces_slug_key" ON "workspaces"("slug");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
