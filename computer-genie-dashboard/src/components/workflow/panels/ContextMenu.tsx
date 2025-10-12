import React, { useEffect, useRef, useCallback } from 'react';
import {
  Copy,
  Scissors,
  Clipboard,
  Trash2,
  Edit3,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Play,
  Pause,
  Square,
  RotateCcw,
  Settings,
  Info,
  Link,
  Unlink,
  Group,
  Ungroup,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Plus,
  Minus,
  Maximize,
  Minimize,
  Bookmark,
  BookmarkX,
  Star,
  StarOff,
  Flag,
  FlagOff,
  MessageSquare,
  Share2,
  Download,
  Upload,
  FileText,
  Code,
  Database,
  Globe,
  Zap,
  Activity,
  BarChart3,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Timer,
  RefreshCw,
  Search,
  Filter,
  SortAsc,
  SortDesc,
  Layout,
  Grid,
  Layers,
  GitBranch,
  Users,
  Bell,
  Tag,
  Folder,
  FolderPlus,
  Archive,
  Trash,
  MoreHorizontal
} from 'lucide-react';

export interface ContextMenuItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  shortcut?: string;
  action: () => void;
  disabled?: boolean;
  danger?: boolean;
  separator?: boolean;
  submenu?: ContextMenuItem[];
}

export interface ContextMenuSection {
  id: string;
  label?: string;
  items: ContextMenuItem[];
}

export interface ContextMenuPosition {
  x: number;
  y: number;
}

interface ContextMenuProps {
  position: ContextMenuPosition | null;
  sections: ContextMenuSection[];
  onClose: () => void;
  className?: string;
}

export const ContextMenu: React.FC<ContextMenuProps> = ({
  position,
  sections,
  onClose,
  className = ''
}) => {
  const menuRef = useRef<HTMLDivElement>(null);
  const [submenuOpen, setSubmenuOpen] = React.useState<string | null>(null);
  const [submenuPosition, setSubmenuPosition] = React.useState<ContextMenuPosition | null>(null);

  // Close menu on outside click or escape
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    if (position) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [position, onClose]);

  // Adjust menu position to stay within viewport
  const getAdjustedPosition = useCallback((pos: ContextMenuPosition, element: HTMLElement) => {
    const rect = element.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    let { x, y } = pos;

    // Adjust horizontal position
    if (x + rect.width > viewportWidth) {
      x = viewportWidth - rect.width - 10;
    }
    if (x < 10) {
      x = 10;
    }

    // Adjust vertical position
    if (y + rect.height > viewportHeight) {
      y = viewportHeight - rect.height - 10;
    }
    if (y < 10) {
      y = 10;
    }

    return { x, y };
  }, []);

  const handleItemClick = useCallback((item: ContextMenuItem, event: React.MouseEvent) => {
    event.stopPropagation();
    
    if (item.disabled) return;
    
    if (item.submenu && item.submenu.length > 0) {
      const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
      setSubmenuPosition({
        x: rect.right + 5,
        y: rect.top
      });
      setSubmenuOpen(item.id);
    } else {
      item.action();
      onClose();
    }
  }, [onClose]);

  const handleSubmenuClose = useCallback(() => {
    setSubmenuOpen(null);
    setSubmenuPosition(null);
  }, []);

  const renderMenuItem = useCallback((item: ContextMenuItem) => {
    if (item.separator) {
      return <div key={item.id} className="h-px bg-gray-700 my-1" />;
    }

    const hasSubmenu = item.submenu && item.submenu.length > 0;
    const isSubmenuOpen = submenuOpen === item.id;

    return (
      <button
        key={item.id}
        onClick={(e) => handleItemClick(item, e)}
        disabled={item.disabled}
        className={`w-full flex items-center justify-between px-3 py-2 text-sm text-left transition-colors ${
          item.disabled
            ? 'text-gray-500 cursor-not-allowed'
            : item.danger
              ? 'text-red-400 hover:bg-red-900/20 hover:text-red-300'
              : 'text-gray-300 hover:bg-gray-700 hover:text-white'
        } ${isSubmenuOpen ? 'bg-gray-700' : ''}`}
      >
        <div className="flex items-center space-x-3">
          {item.icon && (
            <div className="w-4 h-4 flex items-center justify-center">
              {item.icon}
            </div>
          )}
          <span>{item.label}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          {item.shortcut && (
            <span className="text-xs text-gray-500 font-mono">
              {item.shortcut}
            </span>
          )}
          {hasSubmenu && (
            <ArrowRight className="w-3 h-3 text-gray-400" />
          )}
        </div>
      </button>
    );
  }, [submenuOpen, handleItemClick]);

  const renderSubmenu = useCallback(() => {
    if (!submenuOpen || !submenuPosition) return null;

    const openItem = sections
      .flatMap(section => section.items)
      .find(item => item.id === submenuOpen);

    if (!openItem?.submenu) return null;

    return (
      <div
        className="fixed bg-gray-800 border border-gray-700 rounded-lg shadow-lg py-1 z-[60] min-w-[200px]"
        style={{
          left: submenuPosition.x,
          top: submenuPosition.y
        }}
      >
        {openItem.submenu.map(renderMenuItem)}
      </div>
    );
  }, [submenuOpen, submenuPosition, sections, renderMenuItem]);

  if (!position) return null;

  return (
    <>
      <div
        ref={menuRef}
        className={`fixed bg-gray-800 border border-gray-700 rounded-lg shadow-lg py-1 z-50 min-w-[200px] ${className}`}
        style={{
          left: position.x,
          top: position.y
        }}
      >
        {sections.map((section, sectionIndex) => (
          <div key={section.id}>
            {section.label && (
              <div className="px-3 py-1 text-xs font-medium text-gray-500 uppercase tracking-wide">
                {section.label}
              </div>
            )}
            
            {section.items.map(renderMenuItem)}
            
            {sectionIndex < sections.length - 1 && (
              <div className="h-px bg-gray-700 my-1" />
            )}
          </div>
        ))}
      </div>

      {renderSubmenu()}
    </>
  );
};

// Predefined context menu configurations
export const createNodeContextMenu = (
  nodeId: string,
  nodeType: string,
  nodeState: {
    isSelected: boolean;
    isLocked: boolean;
    isVisible: boolean;
    isBookmarked: boolean;
    isFlagged: boolean;
    canExecute: boolean;
    isRunning: boolean;
  },
  actions: {
    onCopy: () => void;
    onCut: () => void;
    onDelete: () => void;
    onDuplicate: () => void;
    onEdit: () => void;
    onToggleVisibility: () => void;
    onToggleLock: () => void;
    onExecute: () => void;
    onPause: () => void;
    onStop: () => void;
    onReset: () => void;
    onGroup: () => void;
    onUngroup: () => void;
    onMoveToFront: () => void;
    onMoveToBack: () => void;
    onToggleBookmark: () => void;
    onToggleFlag: () => void;
    onAddComment: () => void;
    onViewProperties: () => void;
    onExportNode: () => void;
  }
): ContextMenuSection[] => [
  {
    id: 'basic',
    items: [
      {
        id: 'copy',
        label: 'Copy',
        icon: <Copy className="w-4 h-4" />,
        shortcut: 'Ctrl+C',
        action: actions.onCopy
      },
      {
        id: 'cut',
        label: 'Cut',
        icon: <Scissors className="w-4 h-4" />,
        shortcut: 'Ctrl+X',
        action: actions.onCut
      },
      {
        id: 'duplicate',
        label: 'Duplicate',
        icon: <Copy className="w-4 h-4" />,
        shortcut: 'Ctrl+D',
        action: actions.onDuplicate
      },
      {
        id: 'separator1',
        label: '',
        separator: true,
        action: () => {}
      }
    ]
  },
  {
    id: 'execution',
    label: 'Execution',
    items: nodeState.canExecute ? [
      {
        id: 'execute',
        label: nodeState.isRunning ? 'Pause' : 'Execute',
        icon: nodeState.isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />,
        shortcut: 'F5',
        action: nodeState.isRunning ? actions.onPause : actions.onExecute
      },
      {
        id: 'stop',
        label: 'Stop',
        icon: <Square className="w-4 h-4" />,
        shortcut: 'Shift+F5',
        action: actions.onStop,
        disabled: !nodeState.isRunning
      },
      {
        id: 'reset',
        label: 'Reset',
        icon: <RotateCcw className="w-4 h-4" />,
        action: actions.onReset
      }
    ] : []
  },
  {
    id: 'organization',
    label: 'Organization',
    items: [
      {
        id: 'group',
        label: 'Group',
        icon: <Group className="w-4 h-4" />,
        shortcut: 'Ctrl+G',
        action: actions.onGroup
      },
      {
        id: 'ungroup',
        label: 'Ungroup',
        icon: <Ungroup className="w-4 h-4" />,
        shortcut: 'Ctrl+Shift+G',
        action: actions.onUngroup
      },
      {
        id: 'arrange',
        label: 'Arrange',
        icon: <Layers className="w-4 h-4" />,
        submenu: [
          {
            id: 'bring-to-front',
            label: 'Bring to Front',
            icon: <ArrowUp className="w-4 h-4" />,
            action: actions.onMoveToFront
          },
          {
            id: 'send-to-back',
            label: 'Send to Back',
            icon: <ArrowDown className="w-4 h-4" />,
            action: actions.onMoveToBack
          }
        ]
      }
    ]
  },
  {
    id: 'properties',
    label: 'Properties',
    items: [
      {
        id: 'edit',
        label: 'Edit Properties',
        icon: <Edit3 className="w-4 h-4" />,
        shortcut: 'Enter',
        action: actions.onEdit
      },
      {
        id: 'visibility',
        label: nodeState.isVisible ? 'Hide' : 'Show',
        icon: nodeState.isVisible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />,
        action: actions.onToggleVisibility
      },
      {
        id: 'lock',
        label: nodeState.isLocked ? 'Unlock' : 'Lock',
        icon: nodeState.isLocked ? <Unlock className="w-4 h-4" /> : <Lock className="w-4 h-4" />,
        action: actions.onToggleLock
      },
      {
        id: 'bookmark',
        label: nodeState.isBookmarked ? 'Remove Bookmark' : 'Add Bookmark',
        icon: nodeState.isBookmarked ? <BookmarkX className="w-4 h-4" /> : <Bookmark className="w-4 h-4" />,
        action: actions.onToggleBookmark
      },
      {
        id: 'flag',
        label: nodeState.isFlagged ? 'Remove Flag' : 'Add Flag',
        icon: nodeState.isFlagged ? <FlagOff className="w-4 h-4" /> : <Flag className="w-4 h-4" />,
        action: actions.onToggleFlag
      }
    ]
  },
  {
    id: 'actions',
    label: 'Actions',
    items: [
      {
        id: 'comment',
        label: 'Add Comment',
        icon: <MessageSquare className="w-4 h-4" />,
        action: actions.onAddComment
      },
      {
        id: 'properties',
        label: 'View Properties',
        icon: <Info className="w-4 h-4" />,
        action: actions.onViewProperties
      },
      {
        id: 'export',
        label: 'Export Node',
        icon: <Download className="w-4 h-4" />,
        action: actions.onExportNode
      },
      {
        id: 'separator2',
        label: '',
        separator: true,
        action: () => {}
      },
      {
        id: 'delete',
        label: 'Delete',
        icon: <Trash2 className="w-4 h-4" />,
        shortcut: 'Delete',
        action: actions.onDelete,
        danger: true
      }
    ]
  }
];

export const createCanvasContextMenu = (
  canvasState: {
    hasSelection: boolean;
    clipboardHasContent: boolean;
    canUndo: boolean;
    canRedo: boolean;
    showGrid: boolean;
    isLocked: boolean;
  },
  actions: {
    onPaste: () => void;
    onSelectAll: () => void;
    onDeselectAll: () => void;
    onUndo: () => void;
    onRedo: () => void;
    onAddNode: (type: string) => void;
    onToggleGrid: () => void;
    onToggleLock: () => void;
    onZoomToFit: () => void;
    onZoomReset: () => void;
    onExportWorkflow: () => void;
    onImportWorkflow: () => void;
  }
): ContextMenuSection[] => [
  {
    id: 'clipboard',
    items: [
      {
        id: 'paste',
        label: 'Paste',
        icon: <Clipboard className="w-4 h-4" />,
        shortcut: 'Ctrl+V',
        action: actions.onPaste,
        disabled: !canvasState.clipboardHasContent
      },
      {
        id: 'separator1',
        label: '',
        separator: true,
        action: () => {}
      }
    ]
  },
  {
    id: 'selection',
    items: [
      {
        id: 'select-all',
        label: 'Select All',
        icon: <Layout className="w-4 h-4" />,
        shortcut: 'Ctrl+A',
        action: actions.onSelectAll
      },
      {
        id: 'deselect-all',
        label: 'Deselect All',
        icon: <X className="w-4 h-4" />,
        shortcut: 'Ctrl+D',
        action: actions.onDeselectAll,
        disabled: !canvasState.hasSelection
      }
    ]
  },
  {
    id: 'history',
    label: 'History',
    items: [
      {
        id: 'undo',
        label: 'Undo',
        icon: <ArrowLeft className="w-4 h-4" />,
        shortcut: 'Ctrl+Z',
        action: actions.onUndo,
        disabled: !canvasState.canUndo
      },
      {
        id: 'redo',
        label: 'Redo',
        icon: <ArrowRight className="w-4 h-4" />,
        shortcut: 'Ctrl+Y',
        action: actions.onRedo,
        disabled: !canvasState.canRedo
      }
    ]
  },
  {
    id: 'add',
    label: 'Add Node',
    items: [
      {
        id: 'add-trigger',
        label: 'Trigger',
        icon: <Zap className="w-4 h-4" />,
        action: () => actions.onAddNode('trigger')
      },
      {
        id: 'add-action',
        label: 'Action',
        icon: <Play className="w-4 h-4" />,
        action: () => actions.onAddNode('action')
      },
      {
        id: 'add-condition',
        label: 'Condition',
        icon: <GitBranch className="w-4 h-4" />,
        action: () => actions.onAddNode('condition')
      },
      {
        id: 'add-group',
        label: 'Group',
        icon: <Group className="w-4 h-4" />,
        action: () => actions.onAddNode('group')
      }
    ]
  },
  {
    id: 'view',
    label: 'View',
    items: [
      {
        id: 'toggle-grid',
        label: canvasState.showGrid ? 'Hide Grid' : 'Show Grid',
        icon: <Grid className="w-4 h-4" />,
        action: actions.onToggleGrid
      },
      {
        id: 'zoom-to-fit',
        label: 'Zoom to Fit',
        icon: <Maximize className="w-4 h-4" />,
        shortcut: 'Ctrl+0',
        action: actions.onZoomToFit
      },
      {
        id: 'zoom-reset',
        label: 'Reset Zoom',
        icon: <RefreshCw className="w-4 h-4" />,
        shortcut: 'Ctrl+1',
        action: actions.onZoomReset
      }
    ]
  },
  {
    id: 'workflow',
    label: 'Workflow',
    items: [
      {
        id: 'export',
        label: 'Export Workflow',
        icon: <Download className="w-4 h-4" />,
        action: actions.onExportWorkflow
      },
      {
        id: 'import',
        label: 'Import Workflow',
        icon: <Upload className="w-4 h-4" />,
        action: actions.onImportWorkflow
      },
      {
        id: 'lock',
        label: canvasState.isLocked ? 'Unlock Canvas' : 'Lock Canvas',
        icon: canvasState.isLocked ? <Unlock className="w-4 h-4" /> : <Lock className="w-4 h-4" />,
        action: actions.onToggleLock
      }
    ]
  }
];