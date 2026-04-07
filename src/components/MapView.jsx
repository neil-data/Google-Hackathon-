import { Fragment } from 'react'
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, Tooltip } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

function getRiskColor(score) {
  if (score >= 80) return '#dc5f4b'
  if (score >= 40) return '#f2a93b'
  return '#36a46f'
}

export default function MapView({ shipments }) {
  return (
    <MapContainer
      center={[22, 8]}
      zoom={2}
      minZoom={2}
      scrollWheelZoom={false}
      className="map-canvas"
      worldCopyJump
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {shipments.map((shipment) => {
        const color = getRiskColor(shipment.risk)

        return (
          <Fragment key={shipment.id}>
            <Polyline
              positions={[shipment.origin, shipment.destination]}
              pathOptions={{
                color,
                weight: 3,
                opacity: 0.85,
                dashArray: shipment.risk >= 80 ? '8 10' : undefined,
              }}
            >
              <Tooltip sticky>{shipment.id}</Tooltip>
            </Polyline>

            <CircleMarker
              center={shipment.origin}
              radius={7}
              pathOptions={{
                color: '#f6f1e8',
                weight: 2,
                fillColor: color,
                fillOpacity: 1,
              }}
            >
              <Popup>
                <strong>{shipment.id}</strong>
                <br />
                Origin: {shipment.originLabel}
                <br />
                Cargo: {shipment.cargo}
              </Popup>
            </CircleMarker>

            <CircleMarker
              center={shipment.destination}
              radius={5}
              pathOptions={{
                color,
                weight: 2,
                fillColor: '#102033',
                fillOpacity: 1,
              }}
            >
              <Popup>
                <strong>{shipment.id}</strong>
                <br />
                Destination: {shipment.destinationLabel}
                <br />
                ETA: {shipment.eta}
              </Popup>
            </CircleMarker>
          </Fragment>
        )
      })}
    </MapContainer>
  )
}
