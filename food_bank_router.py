from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET


@dataclass
class Route:
    name: str  # human readable route name
    lat: Optional[float] = None  # optional route center latitude
    lon: Optional[float] = None  # optional route center longitude

    @staticmethod
    def load_from_csv(csv_path: str | Path) -> Dict[str, "Route"]:
        routes: Dict[str, Route] = {}  # keyed by route name for quick lookup
        path = Path(csv_path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("region") or "").strip()  # region label from file
                if not name:
                    continue
                lat = Route._parse_float(row.get("lat"))  # parsed latitude value
                lon = Route._parse_float(row.get("lon"))  # parsed longitude value
                routes[name] = Route(name=name, lat=lat, lon=lon)
        return routes

    @staticmethod
    def _parse_float(value: Optional[str]) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None


@dataclass
class Order:
    id: Optional[int] = None  # unique order identifier when available
    name: str = ""  # customer name
    phone: str = ""  # customer phone number
    email: str = ""  # customer email address
    address1: str = ""  # primary street address line
    address2: str = ""  # secondary address details (suite, notes)
    delivery_instructions: str = ""  # additional delivery directions
    lat: Optional[float] = None  # address latitude
    lon: Optional[float] = None  # address longitude
    route: Optional[Route] = None  # assigned route metadata
    route_position: Optional[int] = None  # sequence slot inside the route

    def address_cell(self) -> str:
        if self.address2:
            return f"{self.address1}\n{self.address2}"
        return self.address1


class OrderBook:
    WORKING_HEADERS = [
        "id",
        "name",
        "phone",
        "email",
        "address1",
        "address2",
        "delivery_instructions",
        "lat",
        "lon",
        "route",
        "route_position",
    ]

    def __init__(self, orders: Optional[Iterable[Order]] = None):
        self.orders: List[Order] = list(orders or [])  # normalized list of orders

    @classmethod
    def from_external_csv(
        cls,
        csv_path: str | Path,
        routes: Optional[Dict[str, Route]] = None,
    ) -> "OrderBook":
        path = Path(csv_path)
        orders: List[Order] = []  # collected orders parsed from file
        position_map: Dict[str, int] = {}  # tracks shared route positions per address
        position_counter = 0  # next route position to assign

        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader, start=1):
                address_raw = (row.get("Address") or "").strip()
                address_parts = [piece.strip() for piece in address_raw.splitlines() if piece.strip()]  # split address into lines
                address1 = address_parts[0] if address_parts else ""
                address2 = address_parts[1] if len(address_parts) > 1 else ""

                key = address1.lower()  # normalized address key used for grouping
                if key and key in position_map:
                    route_position = position_map[key]
                else:
                    position_counter += 1
                    route_position = position_counter
                    if key:
                        position_map[key] = route_position

                route_name = (row.get("Route") or "").strip()
                route_obj = cls._resolve_route(route_name, routes)

                orders.append(
                    Order(
                        id=idx,
                        name=(row.get("Name") or "").strip(),
                        phone=(row.get("Phone") or "").strip(),
                        email=(row.get("Email") or "").strip(),
                        address1=address1,
                        address2=address2,
                        delivery_instructions=(row.get("Delivery Instructions") or "").strip(),
                        route=route_obj,
                        route_position=route_position,
                    )
                )
        return cls(orders)

    @classmethod
    def from_working_csv(
        cls,
        csv_path: str | Path,
        routes: Optional[Dict[str, Route]] = None,
    ) -> "OrderBook":
        path = Path(csv_path)
        orders: List[Order] = []  # collected orders from working file
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not any(row.values()):
                    continue
                route_name = (row.get("route") or "").strip()
                route_obj = cls._resolve_route(route_name, routes)

                orders.append(
                    Order(
                        id=cls._parse_int(row.get("id")),
                        name=(row.get("name") or "").strip(),
                        phone=(row.get("phone") or "").strip(),
                        email=(row.get("email") or "").strip(),
                        address1=(row.get("address1") or "").strip(),
                        address2=(row.get("address2") or "").strip(),
                        delivery_instructions=(row.get("delivery_instructions") or "").strip(),
                        lat=Route._parse_float(row.get("lat")),
                        lon=Route._parse_float(row.get("lon")),
                        route=route_obj,
                        route_position=cls._parse_int(row.get("route_position")),
                    )
                )
        return cls(orders)

    def to_working_csv(self, csv_path: str | Path) -> None:
        path = Path(csv_path)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.WORKING_HEADERS)
            writer.writeheader()
            for order in self.orders:
                writer.writerow(
                    {
                        "id": order.id if order.id is not None else "",
                        "name": order.name,
                        "phone": order.phone,
                        "email": order.email,
                        "address1": order.address1,
                        "address2": order.address2,
                        "delivery_instructions": order.delivery_instructions,
                        "lat": _format_float(order.lat),
                        "lon": _format_float(order.lon),
                        "route": order.route.name if order.route else "",
                        "route_position": order.route_position if order.route_position is not None else "",
                    }
                )

    def to_simple_xlsx(self, xlsx_path: str | Path) -> None:
        try:
            from openpyxl import Workbook
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise RuntimeError("openpyxl is required for exporting xlsx files") from exc

        path = Path(xlsx_path)
        workbook = Workbook()
        workbook.remove(workbook.active)

        headers = ["Address", "Name", "Phone", "Email", "Delivery Instructions"]  # column structure for each sheet
        route_names = self._ordered_route_names()  # ordered list of route sheet names
        used_titles: Dict[str, int] = {}  # tracks sheet name collisions

        for route_name in route_names:
            sheet_title = self._make_sheet_title(route_name, used_titles)
            sheet = workbook.create_sheet(title=sheet_title)
            sheet.append(headers)

            for order in self._orders_for_route(route_name):
                sheet.append(
                    [
                        order.address_cell(),
                        order.name,
                        order.phone,
                        order.email,
                        order.delivery_instructions,
                    ]
                )

        workbook.save(path)

    def to_kml(self, kml_path: str | Path, connect_points: bool = True) -> None:
        missing = [order for order in self.orders if order.lat is None or order.lon is None]  # orders lacking coordinates
        if missing:
            identifiers = ", ".join(str(order.id or order.address1) for order in missing)
            raise ValueError(f"Cannot export KML, missing coordinates for: {identifiers}")

        path = Path(kml_path)
        kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        document = ET.SubElement(kml, "Document")

        for route_name in self._ordered_route_names():
            folder = ET.SubElement(document, "Folder")
            ET.SubElement(folder, "name").text = route_name

            coordinates: List[Tuple[Optional[int], str]] = []  # coordinates paired with their route position
            seen_positions: set[int] = set()  # prevents duplicate stops at same position

            for order in self._orders_for_route(route_name):
                if order.route_position is not None and order.route_position in seen_positions:
                    continue
                if order.route_position is not None:
                    seen_positions.add(order.route_position)

                placemark = ET.SubElement(folder, "Placemark")
                label = "" if order.route_position is None else str(order.route_position)
                ET.SubElement(placemark, "name").text = label
                ET.SubElement(placemark, "description").text = order.address_cell()

                point = ET.SubElement(placemark, "Point")
                coord = f"{_format_float(order.lon)},{_format_float(order.lat)},0"  # KML expects lon,lat,altitude
                ET.SubElement(point, "coordinates").text = coord
                coordinates.append((order.route_position, coord))

            if connect_points:
                ordered = [
                    coord
                    for _, coord in sorted(
                        coordinates,
                        key=lambda item: (
                            item[0] is None,
                            item[0] if item[0] is not None else float("inf"),
                        ),
                    )
                ]
                if len(ordered) >= 2:
                    line = ET.SubElement(folder, "Placemark")
                    ET.SubElement(line, "name").text = "Route"
                    line_string = ET.SubElement(line, "LineString")
                    ET.SubElement(line_string, "coordinates").text = " ".join(ordered)

        tree = ET.ElementTree(kml)
        tree.write(path, encoding="utf-8", xml_declaration=True)

    def _orders_for_route(self, route_name: str) -> List[Order]:
        orders = [order for order in self.orders if self._route_name(order) == route_name]  # orders scoped to the route
        return sorted(
            orders,
            key=lambda order: (
                order.route_position if order.route_position is not None else float("inf"),
                order.id if order.id is not None else 0,
            ),
        )

    def _ordered_route_names(self) -> List[str]:
        names: List[str] = []  # preserves first-seen order of route names
        seen = set()
        for order in self.orders:
            name = self._route_name(order)
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names

    @staticmethod
    def _route_name(order: Order) -> str:
        return order.route.name if order.route else "Unassigned"

    @staticmethod
    def _resolve_route(route_name: str, routes: Optional[Dict[str, Route]]) -> Optional[Route]:
        if not route_name:
            return None
        if routes and route_name in routes:
            return routes[route_name]
        return Route(name=route_name)

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _make_sheet_title(name: str, counts: Dict[str, int]) -> str:
        base = (name or "Unassigned")[:31]  # Excel imposes a 31 character limit
        count = counts.get(base, 0)  # how many times we have used this base name
        title = base if count == 0 else OrderBook._with_suffix(base, count)
        while title in counts:
            count += 1
            title = OrderBook._with_suffix(base, count)
        counts[base] = count
        counts[title] = 0
        return title

    @staticmethod
    def _with_suffix(base: str, count: int) -> str:
        if count <= 0:
            return base
        suffix = f"_{count}"
        limit = max(0, 31 - len(suffix))  # ensure suffix does not exceed Excel limit
        trimmed = base[:limit] if limit else ""
        result = f"{trimmed}{suffix}"
        return result[:31]



def _format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"



    main()


def run_workflow() -> None:
    """Adjust the paths below and run this file in PyCharm to execute the workflow."""
    external_csv = Path("data/real_addresses.csv")
    routes_csv = Path("data/route_centers.csv")  # optional route metadata lookup
    working_csv = Path("output/orders_working.csv")  # internal, full-data CSV output
    simple_xlsx = Path("output/orders_simple.xlsx")  # simplified workbook for sharing
    kml_path = Path("output/orders_route.kml")  # optional Google Earth export

    routes = Route.load_from_csv(routes_csv)

    book = OrderBook.from_external_csv(external_csv, routes)

    book.to_working_csv(working_csv)
    print(f"Saved working CSV to {working_csv}")

    book.to_simple_xlsx(simple_xlsx)
    print(f"Saved simplified XLSX to {simple_xlsx}")

    book.to_kml(kml_path, connect_points=False)
    print(f"Saved KML to {kml_path}")


if __name__ == "__main__":
    run_workflow()
