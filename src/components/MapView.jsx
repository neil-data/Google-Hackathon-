import { MapContainer, TileLayer, CircleMarker } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { mockShipments } from '../data/mockShipments'

function getRiskColor(score) {
    if (score >= 80) return '#E24B4A'   // red
    if (score >= 40) return '#EF9F27'   // amber
    return '#1D9E75'                    // green
}

export default function MapView() {
    return (
        <MapContainer center={[20, 0]} zoom={2} style={{ height: '100vh', width: '100%' }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            {mockShipments.map(ship => (
                <CircleMarker
                    key={ship.id}
                    center={ship.origin}
                    radius={8}
                    pathOptions={{ color: getRiskColor(ship.risk), fillColor: getRiskColor(ship.risk), fillOpacity: 0.85 }}
                />
            ))}
        </MapContainer>
    )
}