import { memo, type ReactNode } from "react";

export interface Column<Row> {
  header: string;
  align?: "left" | "right" | "center";
  render: (row: Row) => ReactNode;
}

interface Props<Row> {
  columns: Column<Row>[];
  rows: Row[];
  rowKey: (row: Row, i: number) => string;
}

function DataTable<Row>({ columns, rows, rowKey }: Props<Row>) {
  return (
    <div className="overflow-x-auto rounded-card border border-hairline bg-surface">
      <table className="w-full text-[14px]">
        <thead>
          <tr className="border-b border-hairline">
            {columns.map((c) => (
              <th
                key={c.header}
                className={`px-4 py-3 text-[12px] font-semibold tracking-[0.04em] text-secondary uppercase text-${c.align ?? "left"}`}
                style={{ textAlign: c.align ?? "left" }}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={rowKey(row, i)}
              className="border-b border-hairline last:border-0 hover:bg-white/[0.03]"
            >
              {columns.map((c) => (
                <td
                  key={c.header}
                  className="tnum px-4 py-2.5 text-body"
                  style={{ textAlign: c.align ?? "left" }}
                >
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default memo(DataTable) as typeof DataTable;
